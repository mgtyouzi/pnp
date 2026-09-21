from checkpoint_schedule import checkpoint_label, should_save_checkpoint


def test_saves_every_ten_logical_epochs() -> None:
    saved = [
        checkpoint_label(epoch)
        for epoch in range(100)
        if should_save_checkpoint(epoch, 10)
    ]
    assert saved == list(range(10, 101, 10))


def test_disabled_interval_never_saves() -> None:
    assert not any(should_save_checkpoint(epoch, 0) for epoch in range(100))


def main() -> None:
    test_saves_every_ten_logical_epochs()
    test_disabled_interval_never_saves()
    print("Checkpoint schedule tests passed")


if __name__ == "__main__":
    main()
