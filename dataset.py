
import os
import sys
import json
import albumentations as A
import time
import math

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




def img_loader(dataset, num_classes, phase):
    cell_classes =['0'] # 0,1,2,3

    keys = ['image', 'keypoints'] + [f'keypoints{i}' for i in range(1, num_classes)]
    root = _dataset_root(dataset)
    img_dir, pnt_dir = os.path.join(root, f'{phase}_image'), os.path.join(root, f'{phase}_point')
    data = []
    files = []
    reader = tqdm(os.listdir(img_dir), file=sys.stdout)
    time_string = time.strftime('[%D-%H:%M:%S]',time.localtime())
    reader.set_description(f"{time_string} loading {phase} data")
    cnt =0
    n1, n2, n3, n4 = 0, 0, 0, 0
    for file in reader:
        files.append(os.path.join(img_dir, file))
        data.append(os.path.join(pnt_dir, f"{os.path.splitext(file)[0]}.json"))

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
        # self.index = self.index + 1
        # index = self.index
        # print("index1: ", index)
        index = index % len(self.data)
        assert index <= len(self), 'index range error'
        try:
            sample = self.read_data(self.data[index], self.files[index])
            sample = self.data_transform(sample)
        except:
            # print(self.files[index], " size error!")
            sample = self.__getitem__(index+1)
        # crop失败处理
        # if self.phase == "train" and (sample[0].shape[1] != 1080 or sample[0].shape[2] != 1080):
        if self.phase == "train" and (sample[0].shape[1] != 1024 or sample[0].shape[2] != 1024):
            print(self.files[index], " crop size error!")
            sample = self.__getitem__(index+1)
        # print("index2: ", index)
        return sample


    def read_data(self, data, files):
        # cell_classes = ['浆细胞', '淋巴细胞', '嗜酸性粒细胞', '中性粒细胞']  # 0,1,2,3
        # cell_classes = ['中性粒细胞', '嗜酸性粒细胞', '浆细胞', '淋巴细胞'] # 0,1,2,3
        # cell_classes =['中性粒细胞'] # 0
        # cell_classes = ['阴性细胞', '阳性细胞']  # 0,1
        cell_classes = ['0']  # 0
        # cell_classes = ['0', '1', '2', '3', '4', '5']  # her2
        keys = ['image', 'keypoints'] + [f'keypoints{i}' for i in range(1, self.num_classes)]
        values = [io.imread(files)]

        # assert values[0].shape[0] == 1080 and values[0].shape[1] == 1920, 'size error!'

        with open(data, encoding='utf-8') as f:
            annotations = json.loads(f.read())
            # values += [np.array(annotations[c]).reshape(-1, 2) for c in annotations['classes']]
            temp_list = [np.array(annotations[c]).reshape(-1, 2) for c in cell_classes]

            # # 处理越界点
            # for i in temp_list:
            #     i[:, 0] = np.clip(i[:, 0], 1, 1919)
            #     i[:, 1] = np.clip(i[:, 1], 1, 1079)

            values += temp_list
            # print(values[1:])
            # print(values[1].shape,values[2].shape,values[3].shape,values[4].shape)
        sample = dict(zip(keys, values))
        return sample


def build_dataset(args, image_set):
    mean_std_path = _resolve_mean_std_path(args, image_set)
    if not os.path.exists(mean_std_path):
        raise FileNotFoundError(f"mean/std file not found: {mean_std_path}")
    mean, std = np.load(mean_std_path)
    print(f"[Dataset-{image_set}] dataset={args.dataset}, mean_std={mean_std_path}, mean={mean}, std={std}", flush=True)
    additional_targets = {}
    for i in range(1, args.num_classes):
        additional_targets.update({'keypoints%d' % i: 'keypoints'})
    if image_set == 'train':
        augmentor = A.Compose([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            # A.RandomCrop(height=1080, width=1080, always_apply=True),
            A.ShiftScaleRotate(scale_limit=0.3, rotate_limit=0, shift_limit=0, border_mode=0, value=0, p=0.5),
            # A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, always_apply=False, p=0.5)
            A.RandomCrop(height=1024, width=1024, always_apply=True),
        ], p=1, keypoint_params=A.KeypointParams(format='xy'), additional_targets=additional_targets)
        transform = Preprocessing(mean, std, augmentor)
    elif image_set == 'test':
        transform = Preprocessing(mean, std)
    else:
        raise NotImplementedError
    data_folder = DataFolder(args.dataset, args.num_classes, image_set, transform, mean_std_path=mean_std_path)
    return data_folder

