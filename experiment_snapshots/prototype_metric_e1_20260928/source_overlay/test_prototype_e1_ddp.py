import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from models.prototype_metric_adapter import PrototypeMetricAdapter


class TinyE1Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.detector = nn.Linear(4, 8)
        self.proto_metric_adapter = PrototypeMetricAdapter(8, 6, 4)

    def forward(self, values):
        return {"cls_features": self.detector(values)}


def _worker(rank, world_size, init_file):
    dist.init_process_group(
        "gloo",
        init_method="file://" + init_file,
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(7)
        model = TinyE1Model()
        ddp_model = DDP(model)
        optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
        values = torch.full((3, 4), float(rank + 1))
        output = ddp_model(values)
        proto_output = ddp_model.module.proto_metric_adapter(
            output["cls_features"]
        )
        loss = output["cls_features"].square().mean() + 0.1 * (
            proto_output["margin"].square().mean()
        )
        loss.backward()
        for parameter in ddp_model.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
        optimizer.step()
    finally:
        dist.destroy_process_group()


def main():
    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable")
    with tempfile.TemporaryDirectory() as temp_dir:
        init_file = os.path.join(temp_dir, "ddp_init")
        mp.spawn(_worker, args=(2, init_file), nprocs=2, join=True)
    print("Prototype E1 DDP integration test passed")


if __name__ == "__main__":
    main()
