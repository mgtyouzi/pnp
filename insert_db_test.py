import torch
import argparse
import numpy as np

from skimage import io

from models.detr import build_model
from torchvision import transforms

import json
from employ.utils.db_manager import SliceDBManager
from employ.utils.db_object import Mark

from employ.utils.id_generator import get_guid


@torch.no_grad()
def predict(model, images):
    h, w = images.shape[-2:]
    outputs = model(images)

    points = outputs['pnt_coords'][0].cpu().numpy()
    scores = torch.softmax(outputs['cls_logits'][0], dim=-1).cpu().numpy()

    cross_border_index = (points[:, 0] < 0) | (points[:, 0] >= w) | (points[:, 1] < 0) | (points[:, 1] >= h)
    points = points[~cross_border_index]
    scores = scores[~cross_border_index]

    classes = np.argmax(scores, axis=-1)
    reserved_index = classes < 4
    return deduplicate(points[reserved_index], scores[reserved_index], 10)


def deduplicate(points, scores, interval):
    n = len(points)
    fused = np.full(n, False)
    result = np.zeros((0, 2))
    classes = np.array([])
    for i in range(n):
        if not fused[i]:
            fused_index = np.where(np.linalg.norm(points[[i]] - points[i:], 2, axis=1) < interval)[0] + i
            fused[fused_index] = True

            r_, c_ = np.where(scores[fused_index] == np.max(scores[fused_index]))
            r_, c_ = [r_[0]], [c_[0]]
            result = np.append(result, points[fused_index[r_]], axis=0)
            classes = np.append(classes, c_)
    return result, classes


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs.",
        default=None,
        nargs='+',
    )

    # * Model
    parser.add_argument('--num_classes', type=int, default=4, help="Number of cell categories")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--row', default=2, type=int, help="number of anchor points per row")
    parser.add_argument('--col', default=2, type=int, help="number of anchor points per column")

    return parser


if __name__ == '__main__':
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    multi_class_map_dict = {
        'pdl1_negative_norm': 1,
        'pdl1_negative_tumor': 0,
        'pdl1_positive_norm': 3,
        'pdl1_positive_tumor': 2,
    }

    parser = get_args_parser()
    args = parser.parse_args()

    model = build_model(args)
    ckpt = torch.load('./pdl1_overall', map_location='cpu')
    model.load_state_dict(ckpt['model'])

    model.cuda()
    model.eval()

    mean, std = np.load('./datasets/kyx_data/mean_std.npy')
    slide_path = './PDL1/result/40801-F014-IHC/7/image_7.png'

    dbm = SliceDBManager('slice.db', slide_path=slide_path, ai_type='pdl1')

    dbm.mark_table_name = "Mark_label_pdl1"
    dbm.mark_to_tile_table_name = "MarkToTile_label_pdl1"

    trans = transforms.Compose([
        transforms.ToTensor(),
    ])

    device = 0
    image = io.imread(slide_path)
    image = trans(image)
    for t, m, s in zip(image, mean, std):
        t.sub_(m).div_(s)
    image = image[None].cuda()

    center_coords, cls_labels = predict(model, image)

    type_color_dict = {0: '#00FF00', 1: '#0000FF', 2: '#FF0000', 3: '#FFFF00'}
    group_id_dict = {0: 320, 1: 322, 2: 55, 3: 321}
    mpp = 0.242042

    insert_mark_list = []
    for center_coord, cell_type in zip(center_coords, cls_labels.astype(int).tolist()):
        this_mark = Mark(path=json.dumps({'x': [center_coord[0]], 'y': [center_coord[1]]}),
                         fillColor=type_color_dict[cell_type],
                         mark_type=2,
                         diagnosis=json.dumps({'type': cell_type}),
                         radius=1 / mpp,
                         editable=1,
                         group_id=group_id_dict[cell_type])

        insert_mark_list.append(this_mark.val())

    dbm.insert_point_result_with_count(np.array(center_coords), np.array(cls_labels.astype(int)),
                                       insert_mark_list)
