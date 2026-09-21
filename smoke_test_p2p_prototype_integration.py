import torch

import train_p2p as p2p
from loss import build_criterion
from matcher import build_matcher
from models.detr import build_model


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the original P2P implementation")

    parser = p2p.get_args_parser()
    args = parser.parse_args([
        "--num_classes=1",
        "--backbone=resnet50",
        "--position_embedding=sine",
        "--enc_layers=6",
        "--dim_feedforward=2048",
        "--hidden_dim=256",
        "--dropout=0.1",
        "--nheads=8",
        "--row=2",
        "--col=2",
        "--eos_coef=0.5",
        "--reg_loss_coef=0.002",
        "--cls_loss_coef=1.0",
        "--set_cost_point=0.1",
        "--set_cost_class=1.0",
        "--proto_enable",
        "--proto_embedding_dim=128",
        "--proto_num_fg=2",
        "--proto_num_bg=4",
        "--proto_fg_queue_size=64",
        "--proto_bg_queue_size=64",
        "--proto_temperature=0.2",
        "--proto_loss_weight=0.01",
        # Candidate-radius behavior is covered by test_p2p_prototype.py.  The
        # integration smoke only verifies model/loss/backprop wiring, so keep
        # synthetic Hungarian matches from being rejected after a second
        # stochastic train-mode forward.
        "--proto_initial_positive_radius=1000000",
        "--proto_positive_radius=1000000",
        "--proto_background_radius=30",
        "--proto_max_pos_per_image=64",
        "--proto_max_bg_per_image=16",
        "--proto_max_hard_bg_per_image=16",
        "--proto_max_random_bg_per_image=0",
        "--proto_kmeans_iterations=5",
        "--proto_momentum=0.0",
        "--proto_dead_patience=3",
        "--proto_input_grad_scale=0.1",
        "--proto_debug_interval=1",
        "--proto_inference_fusion=0",
        "--proto_fusion_alpha=0.1",
        "--proto_fusion_clip=2.0",
    ])
    p2p.args = args

    model = build_model(args).cuda(0)
    matcher = build_matcher(args)
    criterion = build_criterion(0, matcher, args)
    prototype = model.prototype_head
    images = torch.randn(1, 3, 256, 256, device="cuda:0")
    model.train()
    seed_outputs = model(images)
    gt_points = seed_outputs["pnt_coords"][0, :4].detach().clone()
    targets = {
        "gt_nums": [4],
        "gt_points": [gt_points],
        "gt_labels": [torch.zeros(4, dtype=torch.long, device="cuda:0")],
        "cell_ratios": torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda:0"),
    }

    prototype.begin_epoch()
    outputs = model(images)
    indices = matcher(outputs, targets)
    base_losses = criterion(outputs, targets, indices=indices)
    proto_loss, diagnostics = prototype.compute_loss_and_cache(
        outputs["proto_embeddings"],
        outputs["pnt_coords"],
        outputs["raw_cls_logits"],
        targets,
        indices,
        anchor_points=outputs["anchor_points"],
    )
    assert proto_loss.item() == 0.0
    initialization_diagnostics = prototype.finalize_epoch()
    assert prototype.prototype_ready.item() == 1, (
        "prototype smoke initialization failed: "
        f"step={diagnostics}, epoch={initialization_diagnostics}"
    )

    prototype.begin_epoch()
    model.zero_grad(set_to_none=True)
    outputs = model(images)
    debug_tensors = model.get_prototype_debug_tensors()
    assert debug_tensors is not None
    debug_tensors["cls_features"].retain_grad()
    indices = matcher(outputs, targets)
    base_losses = criterion(outputs, targets, indices=indices)
    proto_loss, diagnostics = prototype.compute_loss_and_cache(
        outputs["proto_embeddings"],
        outputs["pnt_coords"],
        outputs["raw_cls_logits"],
        targets,
        indices,
        anchor_points=outputs["anchor_points"],
    )
    total = base_losses.sum() + args.proto_loss_weight * proto_loss
    gradient_diagnostics = p2p._gradient_pair_metrics(
        base_losses[1],
        args.proto_loss_weight * proto_loss,
        [("cls_features", debug_tensors["cls_features"])],
    )
    total.backward()
    update_diagnostics = prototype.finalize_epoch()

    assert outputs["pnt_coords"].shape[:2] == outputs["raw_cls_logits"].shape[:2]
    assert outputs["proto_embeddings"].shape[-1] == args.proto_embedding_dim
    assert proto_loss.item() > 0.0
    assert torch.isfinite(total).item()
    assert debug_tensors["cls_features"].grad is not None
    assert gradient_diagnostics["gradient_cls_features_available"] == 1.0
    assert gradient_diagnostics["gradient_cls_features_primary_available"] == 1.0
    assert gradient_diagnostics["gradient_cls_features_auxiliary_available"] == 1.0
    assert gradient_diagnostics["gradient_cls_features_proto_norm"] > 0.0
    projector_grad = sum(
        float(parameter.grad.detach().abs().sum().item())
        for parameter in prototype.projector.parameters()
        if parameter.grad is not None
    )
    assert projector_grad > 0.0

    model.eval()
    with torch.no_grad():
        inference = model(images)
    assert torch.equal(inference["cls_logits"], inference["raw_cls_logits"])
    assert torch.isfinite(inference["cls_logits"]).all().item()

    print("P2P prototype integration smoke test passed")
    print("step diagnostics =", diagnostics)
    print("gradient diagnostics =", gradient_diagnostics)
    print("initialization diagnostics =", initialization_diagnostics)
    print("update diagnostics =", update_diagnostics)


if __name__ == "__main__":
    main()
