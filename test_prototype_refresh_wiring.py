from pathlib import Path


ROOT = Path(__file__).resolve().parent


def test_refresh_runs_after_backward_and_optimizer_step():
    source = (ROOT / "train_p2p.py").read_text(encoding="utf-8")
    backward = source.index("loss.backward()")
    optimizer_step = source.index("optimizer.step()", backward)
    refresh = source.index("refresh_diag = prototype_head.maybe_refresh_prototypes()")
    assert backward < optimizer_step < refresh


def test_current_projector_refresh_receives_unprojected_source_features():
    model_source = (ROOT / "models" / "detr.py").read_text(encoding="utf-8")
    train_source = (ROOT / "train_p2p.py").read_text(encoding="utf-8")
    assert "outputs['proto_source_features'] = cls_features" in model_source
    assert "self.training and self.prototype_head.refresh_interval_steps > 0" in model_source
    assert "source_features=outputs.get('proto_source_features')" in train_source


def test_v24_entry_enables_window_refresh():
    source = (ROOT / "run_p2p_prototype_v24_paired.sh").read_text(
        encoding="utf-8"
    )
    assert 'PROTO_REFRESH_INTERVAL="${PROTO_REFRESH_INTERVAL:-100}"' in source
    assert '--proto_refresh_interval_steps="${PROTO_REFRESH_INTERVAL}"' in source
    assert "prototype_v2_4_window_refresh_20260816" in source


def main():
    test_refresh_runs_after_backward_and_optimizer_step()
    test_current_projector_refresh_receives_unprojected_source_features()
    test_v24_entry_enables_window_refresh()
    print("Prototype-v2.4 refresh wiring tests passed")


if __name__ == "__main__":
    main()
