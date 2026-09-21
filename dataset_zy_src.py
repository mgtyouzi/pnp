

import os
import sys
import json
import albumentations as A
import time
import math
import numpy as np
from tqdm import tqdm
from skimage import io
from transforms import *
from torch.utils.data import Dataset
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def _dataset_root(dataset):
    return dataset if os.path.isabs(dataset) else os.path.join('.', 'datasets', dataset)


def _resolve_mean_std_path(args, image_set):
    phase_path = getattr(args, f"{image_set}_mean_std_path", "")
    shared_path = getattr(args, "mean_std_path", "")
    if phase_path:
        return phase_path
    if shared_path:
        return shared_path
    return os.path.join(_dataset_root(args.dataset), "mean_std.npy")


def _resolve_data_phase(args, image_set):
    if image_set == "test":
        return getattr(args, "eval_split", "test")
    return image_set


def img_loader(dataset, num_classes, phase):
    cell_classes = ['印戒细胞']
    keys = ['image', 'keypoints'] + [f'keypoints{i}' for i in range(1, num_classes)]
    root = _dataset_root(dataset)
    img_dir, pnt_dir = os.path.join(root, f'{phase}_image'), os.path.join(root, f'{phase}_point')
    data = []
    files = []
    reader = tqdm(os.listdir(img_dir), file=sys.stdout)
    time_string = time.strftime('[%D-%H:%M:%S]', time.localtime())
    reader.set_description(f"{time_string} loading {phase} data")

    for file in reader:
        image_path = os.path.join(img_dir, file)
        base_name = os.path.splitext(file)[0]
        json_file = f"{base_name}.jpg.json"
        json_path = os.path.join(pnt_dir, json_file)

        if not os.path.exists(json_path):
            print(f"Warning: JSON file not found - {json_path}")
            continue

        files.append(image_path)
        data.append(json_path)

    return data, files


class DataFolder(Dataset):
    def __init__(self, dataset, num_classes, phase, data_transform, mean_std_path=""):
        self.dataset = dataset
        self.dataset_root = _dataset_root(dataset)
        self.num_classes = num_classes
        self.phase = phase
        self.mean_std_path = mean_std_path
        self.data, self.files = img_loader(dataset, num_classes, phase)
        self.data_transform = data_transform
        self.max_retries = min(20, max(len(self.data), 1))
        self._debug_counts = {
            "getitem_requests": 0,
            "successful_samples": 0,
            "load_or_transform_errors": 0,
            "replacement_samples": 0,
            "crop_size_errors": 0,
            "source_points_seen": 0,
            "source_points_clipped": 0,
            "source_points_dropped": 0,
            "augmentation_points_input": 0,
            "augmentation_points_output": 0,
            "augmentation_points_removed": 0,
        }
        self._debug_records = []
        self._sample_trace = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        if not self.data:
            raise RuntimeError(f"empty dataset: {self.dataset}/{self.phase}")
        requested_index = index % len(self.data)
        self._debug_counts["getitem_requests"] += 1
        errors = []
        for retry in range(self.max_retries):
            actual_index = (requested_index + retry) % len(self.data)
            image_path = self.files[actual_index]
            try:
                raw_sample = self.read_data(self.data[actual_index], image_path)
                input_points = sum(
                    len(value) for key, value in raw_sample.items() if key.startswith("keypoints")
                )
                sample = self.data_transform(raw_sample)
                output_points = int(sample[1].numel() // 2)
                self._debug_counts["augmentation_points_input"] += input_points
                self._debug_counts["augmentation_points_output"] += output_points
                self._debug_counts["augmentation_points_removed"] += max(
                    input_points - output_points, 0
                )
                if self.phase == "train" and (
                    sample[0].shape[1] != 1080 or sample[0].shape[2] != 1920
                ):
                    self._debug_counts["crop_size_errors"] += 1
                    raise ValueError(
                        f"crop size {tuple(sample[0].shape)} is not Cx1080x1920"
                    )
                self._debug_counts["successful_samples"] += 1
                if len(self._sample_trace) < 128:
                    self._sample_trace.append(image_path)
                if retry > 0:
                    self._debug_counts["replacement_samples"] += 1
                    self._record_debug(
                        "replacement",
                        requested_index=requested_index,
                        actual_index=actual_index,
                        requested_image=self.files[requested_index],
                        actual_image=image_path,
                        retry=retry,
                        preceding_errors=errors,
                    )
                return sample
            except Exception as error:
                self._debug_counts["load_or_transform_errors"] += 1
                error_payload = {
                    "retry": retry,
                    "actual_index": actual_index,
                    "image": image_path,
                    "annotation": self.data[actual_index],
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                errors.append(error_payload)
                self._record_debug("sample_error", **error_payload)
                print(
                    "[Dataset-error] "
                    f"phase={self.phase}, requested={requested_index}, actual={actual_index}, "
                    f"image={image_path}, retry={retry}, error={error!r}",
                    flush=True,
                )
        raise RuntimeError(
            f"failed to load dataset index {requested_index} after {self.max_retries} retries; "
            f"last_errors={errors[-3:]}"
        )

    def _record_debug(self, event, **payload):
        if len(self._debug_records) < 2000:
            self._debug_records.append({"event": event, **payload})

    def get_debug_state(self, reset=False):
        payload = {
            "dataset": self.dataset,
            "phase": self.phase,
            "counts": dict(self._debug_counts),
            "records": list(self._debug_records),
            "sample_trace": list(self._sample_trace),
        }
        if reset:
            for key in self._debug_counts:
                self._debug_counts[key] = 0
            self._debug_records = []
            self._sample_trace = []
        return payload

    def read_data(self, data, files):
       # cell_classes = ['印戒细胞', '\u5370\u6212\u7ec6\u80de']
        cell_classes = ['印戒细胞']
        keys = ['image', 'keypoints'] + [f'keypoints{i}' for i in range(1, self.num_classes)]
        image = io.imread(files)
        values = [image]
        height, width = image.shape[:2]

        with open(data, encoding='utf-8') as f:
            annotations = json.loads(f.read())
            # print("Annotations keys:", annotations.keys())

            # 解析 JSON 数据中的标注点
            points = []
            for ann in annotations.get('annotation', []):
                label = ann['label'][0]
                # 检查标签是否为目标类别
                if any(cls in label for cls in cell_classes):
                    x = float(ann['position']['x'][0])
                    y = float(ann['position']['y'][0])
                    self._debug_counts["source_points_seen"] += 1
                    if not np.isfinite(x) or not np.isfinite(y):
                        self._debug_counts["source_points_dropped"] += 1
                        self._record_debug(
                            "source_point_dropped",
                            image=files,
                            x=x,
                            y=y,
                            reason="non_finite",
                        )
                        continue
                    if self.phase == "train" and not (
                        0.0 <= x < width and 0.0 <= y < height
                    ):
                        tolerance = 2.0
                        if (
                            -tolerance <= x <= width + tolerance
                            and -tolerance <= y <= height + tolerance
                        ):
                            old_x, old_y = x, y
                            x = min(max(x, 0.0), width - 1e-4)
                            y = min(max(y, 0.0), height - 1e-4)
                            self._debug_counts["source_points_clipped"] += 1
                            self._record_debug(
                                "source_point_clipped",
                                image=files,
                                old_x=old_x,
                                old_y=old_y,
                                x=x,
                                y=y,
                            )
                        else:
                            self._debug_counts["source_points_dropped"] += 1
                            self._record_debug(
                                "source_point_dropped",
                                image=files,
                                x=x,
                                y=y,
                                reason="outside_image",
                            )
                            continue
                    points.append([x, y])

            temp_list = [np.array(points).reshape(-1, 2)] if points else [np.empty((0, 2))]
            values += temp_list

        sample = dict(zip(keys, values))
        return sample


def build_dataset(args, image_set):
    phase = _resolve_data_phase(args, image_set)
    mean_std_path = _resolve_mean_std_path(args, image_set)
    if not os.path.exists(mean_std_path):
        raise FileNotFoundError(f"mean/std file not found: {mean_std_path}")
    mean, std = np.load(mean_std_path)
    print(f"[Dataset-{image_set}] phase={phase}, dataset={args.dataset}, mean_std={mean_std_path}, mean={mean}, std={std}", flush=True)
    additional_targets = {}
    for i in range(1, args.num_classes):
        additional_targets.update({'keypoints%d' % i: 'keypoints'})

    if phase == 'train':
        augmentor = A.Compose([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.ShiftScaleRotate(scale_limit=0.3, rotate_limit=0, shift_limit=0, border_mode=0, value=0, p=0.5),
            A.RandomCrop(height=1080, width=1920, always_apply=True),#3.9 1024*1024
        ], p=1, keypoint_params=A.KeypointParams(
            format='xy', remove_invisible=True, check_each_transform=False
        ), additional_targets=additional_targets)
        transform = Preprocessing(mean, std, augmentor)
    elif phase in ('val', 'test'):
        transform = Preprocessing(mean, std)
    else:
        raise NotImplementedError

    data_folder = DataFolder(args.dataset, args.num_classes, phase, transform, mean_std_path=mean_std_path)
    return data_folder



