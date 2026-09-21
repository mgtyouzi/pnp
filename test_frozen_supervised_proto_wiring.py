from pathlib import Path


ROOT = Path(__file__).resolve().parent


def test_training_parser_and_model_expose_new_mode() -> None:
    train = (ROOT / "train_p2p.py").read_text(encoding="utf-8")
    detr = (ROOT / "models" / "detr.py").read_text(encoding="utf-8")
    assert "frozen_supervised_metric" in train
    assert "frozen_supervised_metric" in detr
    assert "--checkpoint_interval" in train


def test_queue_is_single_gpu_sequential_and_100_epochs() -> None:
    queue = (
        ROOT / "run_frozen_supervised_proto_100ep_queue_gpu3.sh"
    ).read_text(encoding="utf-8")
    assert 'GPU="${GPU:-3}"' in queue
    assert 'EPOCHS="${EPOCHS:-100}"' in queue
    assert 'CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10}"' in queue
    assert "torchrun" not in queue
    control = queue.index("CONTROL_TRAIN_START")
    prototype = queue.index("PROTOTYPE_TRAIN_START")
    assert control < prototype


def main() -> None:
    test_training_parser_and_model_expose_new_mode()
    test_queue_is_single_gpu_sequential_and_100_epochs()
    print("Frozen supervised prototype wiring tests passed")


if __name__ == "__main__":
    main()
