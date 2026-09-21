from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    path = ROOT / "run_source_supervised_candidate_proto_paired_gpu6.sh"
    assert path.is_file()
    source = path.read_text(encoding="utf-8")
    assert 'GPU="${GPU:-6}"' in source
    assert "build_source_supervised_candidate_proto_bank.py" in source
    assert "source_supervised_candidate_proto" in source
    assert "CONTROL_DIR" in source and "PROTOTYPE_DIR" in source
    assert "--epochs=10" in source
    assert "--proto_inference_fusion=1" in source
    assert "--proto_loss_weight=1.0" in source
    assert "prototype_raw" in source and "prototype_fused" in source
    assert "decide_source_supervised_candidate_proto_gate.py" in source
    assert "冰冻-2025" not in source
    assert "0318/best_model.pth" in source
    print("Source-supervised candidate prototype run-script tests passed")


if __name__ == "__main__":
    main()
