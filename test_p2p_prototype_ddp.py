import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from models.p2p_prototype import CandidatePrototypeBank


class TinyPrototypeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(4, 4)
        self.cls_head = nn.Linear(4, 2)
        self.prototype_head = CandidatePrototypeBank(
            feat_dim=4,
            embedding_dim=3,
            num_fg_prototypes=1,
            num_bg_prototypes=1,
            input_gradient_scale=0.25,
        )
        self._debug_features = None

    def forward(self, values):
        features = self.shared(values)
        self._debug_features = features
        raw_logits = self.cls_head(features)
        prototype = self.prototype_head(features, raw_logits)
        return raw_logits, prototype["embeddings"]


def assert_pre_ddp_shared_feature_gradients_are_available(local_rank):
    model = TinyPrototypeModel().cuda(local_rank)
    wrapped = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        find_unused_parameters=True,
    )
    values = torch.tensor(
        [[[0.2, -0.3, 0.5, 0.7], [-0.4, 0.8, 0.1, -0.2]]],
        device=f"cuda:{local_rank}",
    )
    raw_logits, embeddings = wrapped(values)
    primary_loss = raw_logits.square().sum()
    auxiliary_loss = (embeddings * torch.tensor(
        [0.3, -0.5, 0.7], device=embeddings.device
    )).sum()
    primary_gradient = torch.autograd.grad(
        primary_loss, model._debug_features, retain_graph=True
    )[0]
    auxiliary_gradient = torch.autograd.grad(
        auxiliary_loss, model._debug_features, retain_graph=True
    )[0]

    assert primary_gradient.norm().item() > 0.0
    assert auxiliary_gradient.norm().item() > 0.0


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")

    module = CandidatePrototypeBank(
        feat_dim=2,
        embedding_dim=2,
        num_fg_prototypes=2,
        num_bg_prototypes=2,
        foreground_queue_size=16,
        background_queue_size=16,
        kmeans_iterations=3,
    ).cuda(local_rank)
    offset = float(dist.get_rank()) * 0.05
    module.begin_epoch()
    module.cache_embeddings(
        torch.tensor(
            [[1.0, offset], [0.9, 0.1 + offset], [0.0, 1.0], [0.1, 0.9]],
            device=f"cuda:{local_rank}",
        ),
        torch.tensor(
            [[-1.0, -offset], [-0.9, -0.1 - offset], [0.0, -1.0], [-0.1, -0.9]],
            device=f"cuda:{local_rank}",
        ),
    )
    diagnostics = module.finalize_epoch()

    assert module.prototype_ready.item() == 1
    for value in (module.fg_prototypes, module.bg_prototypes):
        reference = value.clone()
        dist.broadcast(reference, src=0)
        assert torch.allclose(value, reference, atol=1e-6)

    periodic = CandidatePrototypeBank(
        feat_dim=2,
        embedding_dim=2,
        num_fg_prototypes=1,
        num_bg_prototypes=1,
        foreground_queue_size=8,
        background_queue_size=8,
        prototype_momentum=0.0,
        update_mode="kmeans_ema",
        refresh_interval_steps=1,
    ).cuda(local_rank)
    periodic.begin_epoch()
    periodic.cache_embeddings(
        torch.tensor([[1.0, offset]], device=f"cuda:{local_rank}"),
        torch.tensor([[-1.0, -offset]], device=f"cuda:{local_rank}"),
    )
    refresh = periodic.maybe_refresh_prototypes()
    assert refresh["prototype_refresh_event"] == 1
    assert periodic.prototype_refresh_count.item() == 1
    for value in (periodic.fg_prototypes, periodic.bg_prototypes):
        reference = value.clone()
        dist.broadcast(reference, src=0)
        assert torch.allclose(value, reference, atol=1e-6)

    assert_pre_ddp_shared_feature_gradients_are_available(local_rank)

    if dist.get_rank() == 0:
        print("P2P prototype DDP synchronization and gradient test passed")
        print("diagnostics =", diagnostics)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
