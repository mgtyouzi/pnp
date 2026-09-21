from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "run_gt_foreground_proxy_paired_gpu4.sh"


def main():
    assert SCRIPT.is_file(), "GPU4 paired run script is missing"
    source = SCRIPT.read_text(encoding="utf-8")
    required = [
        'GPU="${GPU:-4}"',
        "--proto_mode=gt_foreground_proxy",
        "--proto_inference_fusion=0",
        "--proto_num_fg=4",
        "--proto_embedding_dim=64",
        "--proto_gt_warmup_epochs=5",
        "--proto_loss_weight=0.02",
        "--proto_positive_radius=15",
        "--proto_background_radius=30",
        "--proto_max_hard_bg_per_image=16",
        "test_gt_foreground_proxy.py",
        "control_p2p",
        "gt_foreground_proxy",
        "diagnose_cross_domain.py",
    ]
    missing = [value for value in required if value not in source]
    assert not missing, missing
    assert "--proto_inference_fusion=1" not in source
    assert 'EVAL_ONLY="${EVAL_ONLY:-0}"' in source
    assert '--mean_std_path "${FROZEN_DATASET}/mean_std.npy"' not in source
    print("GT foreground proxy run-script tests passed")


if __name__ == "__main__":
    main()
