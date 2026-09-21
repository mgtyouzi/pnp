import numpy as np

from export_frozen_supervised_metric_bank import build_groupwise_train_calibration_split


def test_split_is_image_disjoint_and_keeps_every_group() -> None:
    group_index = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    image_index = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    train, calibration = build_groupwise_train_calibration_split(
        group_index, image_index, calibration_fraction=0.5, seed=3
    )
    train_images = set(image_index[train])
    calibration_images = set(image_index[calibration])
    assert train_images.isdisjoint(calibration_images)
    assert set(group_index[train]) == {0, 1}
    assert set(group_index[calibration]) == {0, 1}


def main() -> None:
    test_split_is_image_disjoint_and_keeps_every_group()
    print("Frozen supervised metric bank export tests passed")


if __name__ == "__main__":
    main()
