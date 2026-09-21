def should_save_checkpoint(epoch: int, interval: int) -> bool:
    """Return True at each completed logical interval (epoch is zero based)."""
    return int(interval) > 0 and (int(epoch) + 1) % int(interval) == 0


def checkpoint_label(epoch: int) -> int:
    return int(epoch) + 1
