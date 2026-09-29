import os
import tempfile

import torch
from torch import nn

from prototype_e1_checkpoint import load_baseline_checkpoint


class TinyDetector(nn.Module):
    def __init__(self, with_adapter=True):
        super().__init__()
        self.detector = nn.Linear(3, 2)
        self.proto_metric_adapter = nn.Linear(2, 1) if with_adapter else None


def test_baseline_loads_and_only_adapter_may_be_missing():
    source = TinyDetector(with_adapter=False)
    target = TinyDetector(with_adapter=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = os.path.join(temp_dir, "baseline.pth")
        torch.save({"model": source.state_dict(), "epoch": 42}, checkpoint_path)
        report = load_baseline_checkpoint(target, checkpoint_path)

    assert report["loaded_tensors"] == len(source.state_dict())
    assert report["missing_new_parameters"] == [
        "proto_metric_adapter.bias", "proto_metric_adapter.weight"
    ]
    assert torch.equal(target.detector.weight, source.detector.weight)
    assert torch.equal(target.detector.bias, source.detector.bias)


def test_incomplete_detector_checkpoint_is_rejected():
    source = TinyDetector(with_adapter=False)
    state = source.state_dict()
    state.pop("detector.bias")
    target = TinyDetector(with_adapter=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = os.path.join(temp_dir, "incomplete.pth")
        torch.save({"model": state}, checkpoint_path)
        try:
            load_baseline_checkpoint(target, checkpoint_path)
        except RuntimeError as error:
            assert "did not initialize the full detector" in str(error)
        else:
            raise AssertionError("incomplete baseline initialization was accepted")


def main():
    test_baseline_loads_and_only_adapter_may_be_missing()
    test_incomplete_detector_checkpoint_is_rejected()
    print("Prototype E1 checkpoint tests passed")


if __name__ == "__main__":
    main()
