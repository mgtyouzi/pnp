from pathlib import Path


def main():
    path = Path(__file__).resolve().parent / "run_prototype_consensus_union_gpu.sh"
    source = path.read_text(encoding="utf-8")
    assert 'GPU="${GPU:-0}"' in source
    assert 'CUDA_VISIBLE_DEVICES="${GPU}"' in source
    assert "prototype_consensus_union_audit.py" in source
    assert "test_prototype_consensus_union_audit.py" in source
    assert "--gpu=0" in source
    assert "--prototype_thresholds=0.56:0.72:0.02" in source
    assert "--support_floors=0.05,0.10,0.20,0.30,0.40,0.50" in source
    assert "--agreement_radii=3,5,8,12" in source
    print("Prototype consensus union run-script tests passed")


if __name__ == "__main__":
    main()
