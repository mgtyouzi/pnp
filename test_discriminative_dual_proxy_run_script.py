from pathlib import Path


ROOT = Path(__file__).resolve().parent
GATE = ROOT / "run_discriminative_dual_proxy_gate_gpu.sh"
PAIRED = ROOT / "run_discriminative_dual_proxy_paired.sh"


def test_representation_gate_stops_before_detector_training():
    assert GATE.is_file()
    source = GATE.read_text(encoding="utf-8")
    assert "set -euo pipefail" in source
    assert "--proto_freeze_detector=1" in source
    assert "--proto_debug_interval=0" in source
    assert "--epochs=5" in source
    assert "evaluate_discriminative_dual_proxy_gate.py" in source
    assert "decide_discriminative_dual_proxy_gate.py" in source
    assert "[STOP] representation gate failed; no detector training" in source
    assert "dual_proxy_bank.pth" in source
    assert "--proto_max_hard_bg_per_image=32" in source
    assert "--proto_max_random_bg_per_image=32" in source
    assert "--proto_dual_supcon_weight=0.1" in source


def test_representation_gate_supports_safe_dual_gpu_checkpointing():
    source = GATE.read_text(encoding="utf-8")
    assert 'GPUS="${GPUS:-${GPU}}"' in source
    assert "torchrun" in source
    assert "--nproc_per_node" in source
    assert "--start_eval=999" in source
    assert "--checkpoint_interval=1" in source
    assert "--debug_save_final_checkpoint=1" in source
    assert 'CUDA_VISIBLE_DEVICES="${GPUS}"' in source


def test_representation_gate_can_resume_at_evaluation_without_retraining():
    source = GATE.read_text(encoding="utf-8")
    assert 'EVAL_ONLY="${EVAL_ONLY:-0}"' in source
    assert 'if [ "${EVAL_ONLY}" != "1" ]; then' in source
    assert "[Gate] evaluation-only recovery" in source


def test_paired_launcher_requires_a_passed_bank_and_supports_eval_only():
    assert PAIRED.is_file()
    source = PAIRED.read_text(encoding="utf-8")
    assert "EVAL_ONLY" in source
    assert "dual_proxy_bank.pth" in source
    assert "gate_pass" in source
    assert "detector_init_sha256" in source
    assert "baseline checkpoint does not match the gate bank" in source
    assert "--epochs=10" in source
    assert "--proto_loss_weight=0.02" in source
    assert "--proto_max_hard_bg_per_image=32" in source
    assert "--proto_max_random_bg_per_image=32" in source
    assert "--proto_dual_supcon_weight=0.1" in source
    assert "--proto_dual_separation_weight=1.0" in source
    assert "--proto_dual_balance_weight=0.1" in source
    assert "--proto_inference_fusion=0" in source
    assert "latest_discriminative" not in source


def main():
    test_representation_gate_stops_before_detector_training()
    test_representation_gate_supports_safe_dual_gpu_checkpointing()
    test_representation_gate_can_resume_at_evaluation_without_retraining()
    test_paired_launcher_requires_a_passed_bank_and_supports_eval_only()
    print("Discriminative dual-proxy run-script tests passed")


if __name__ == "__main__":
    main()
