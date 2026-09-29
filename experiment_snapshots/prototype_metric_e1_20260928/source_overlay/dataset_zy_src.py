

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        index = index % len(self.data)
        assert index <= len(self), 'index range error'
        try:
            sample = self.read_data(self.data[index], self.files[index])
            sample = self.data_transform(sample)
        except Exception as e:
            print(f"Error loading data: {e}")
            sample = self.__getitem__(index + 1)

        if self.phase == "train" and (sample[0].shape[1] != 1080 or sample[0].shape[2] != 1920):
            print(self.files[index], " crop size error!")
            sample = self.__getitem__(index + 1)

        return sample

    def read_data(self, data, files):
       # cell_classes = ['印戒细胞', '\u5370\u6212\u7ec6\u80de']
        cell_classes = ['印戒细胞']
        keys = ['image', 'keypoints'] + [f'keypoints{i}' for i in range(1, self.num_classes)]
        values = [io.imread(files)]

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
        ], p=1, keypoint_params=A.KeypointParams(format='xy'), additional_targets=additional_targets)
        transform = Preprocessing(mean, std, augmentor)
    elif phase in ('val', 'test'):
        transform = Preprocessing(mean, std)
    else:
        raise NotImplementedError

    data_folder = DataFolder(args.dataset, args.num_classes, phase, transform, mean_std_path=mean_std_path)
    return data_folder
