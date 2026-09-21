from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    detr = (ROOT / "models" / "detr.py").read_text(encoding="utf-8")
    loader = (ROOT / "paired_detection_threshold_sweep.py").read_text(
        encoding="utf-8"
    )
    script = (ROOT / "run_prototype_image_adaptive_union_gpu.sh").read_text(
        encoding="utf-8"
    )
    assert "prototype_diagnostic_eval" in detr
    assert "collect_proto_embeddings" in loader
    assert 'GPU="${GPU:-7}"' in script
    assert 'CUDA_VISIBLE_DEVICES="${GPU}"' in script
    assert "prototype_image_adaptive_union_audit.py" in script
    assert "--gpu=0" in script
    assert "--foreground_support_threshold=0.8" in script
    assert "--background_support_threshold=0.05" in script
    assert "--num_foreground_prototypes=4" in script
    assert "--num_background_prototypes=4" in script
    print("Prototype image-adaptive structure tests passed")


if __name__ == "__main__":
    main()
