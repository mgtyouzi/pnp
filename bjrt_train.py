import os
import sys
import argparse
import random
import wandb
import cv2 as cv
import time

from utils import *
from tqdm import tqdm

from dataset import build_dataset
from models.detr import build_model
from loss import build_criterion
from matcher import build_matcher
from lr_sched import adjust_learning_rate

import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler


# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '29500'
# os.environ['RANK'] = "0"
# os.environ['WORLD_SIZE'] = "1"

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs.",
        default=None,
        nargs='+',
    )

    # * Optimizer
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # * Train
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--start_eval', default=20, type=int)

    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--output_dir', default='/home/zhangwanqi/zhengdasan_p2p_0121/pth/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--num_classes', type=int, default=4,
                        help="Number of cell categories")

    # * Loss
    parser.add_argument('--reg_loss_coef', default=2e-3, type=float)
    parser.add_argument('--cls_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=1.0, type=float,
                        help="Relative classification weight of the no-object class")

    # * Matcher
    parser.add_argument('--set_cost_point', default=0.1, type=float,
                        help="L2 point coefficient in the matching cost")
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")

    # * Model
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
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

    # * Dataset
    parser.add_argument('--dataset', default='', type=str)
    parser.add_argument('--num_workers', default=0, type=int)

    # * Evaluator
    parser.add_argument('--match_dis', default=15, type=int)

    # * Distributed training
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser


def construct_dataset():
    dataset_train = build_dataset(args, 'train')
    dataset_val = build_dataset(args, 'test')

    if args.distributed:
        train_sampler = DistributedSampler(dataset_train)
        data_loader_train = DataLoader(dataset_train, sampler=train_sampler, batch_size=args.batch_size,
                                       num_workers=0, collate_fn=collate_fn_pad)
    else:
        data_loader_train = DataLoader(dataset_train, shuffle=True, batch_size=args.batch_size,
                                       num_workers=0, collate_fn=collate_fn_pad)
    data_loader_val = DataLoader(dataset_val, shuffle=False, batch_size=1, num_workers=0,
                                 collate_fn=collate_fn_pad)
    data_loaders = {'train': data_loader_train, 'test': data_loader_val}
    return data_loaders


def train():
    if args.distributed:
        rank = args.gpu
        print(rank)
    else:
        rank = 0

    model = build_model(args).cuda(rank)
    model_without_ddp = model

    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)
        model_without_ddp = model.module

    matcher = build_matcher(args)
    criterion = build_criterion(rank, matcher, args)
    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    data_loaders = construct_dataset()
    evaluator = Evaluator(data_loaders['test'], args.num_classes)

    # 初始化最佳性能指标
    best_metric_value = 0
    best_epoch = None

    first_eval = True

    # 加载检查点并初始化 metrics
    metrics = load_checkpoint(args, model_without_ddp, optimizer) if args.resume else {'metrics': [], '分类指标': []}
    max_cls_mf1 = metrics.get('分类F1', 0)

    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, data_loaders['train'], optimizer, args.clip_max_norm, epoch, criterion, rank)

        if trigger_eval(epoch, args.start_eval):
            if first_eval:
                save_model(epoch, args, metrics, model_without_ddp, optimizer, mode='best')
                first_eval = False

            # 计算评估指标
            new_metrics = evaluator.calculate_metrics(model, rank=rank, effective_matching_dis=args.match_dis)

            # 更新 metrics 字典
            if 'metrics' not in metrics:
                metrics['metrics'] = []
            if '分类指标' not in metrics:
                metrics['分类指标'] = []

            metrics['metrics'].append(new_metrics.get('metric_value', None))
            metrics['分类指标'].append(new_metrics.get('cls_mf1', None))

            # 获取当前epoch的评估指标值
            current_metric_value = metrics['metrics'][-1] if metrics['metrics'] else None

            # 打印评估指标
            print(f"Epoch {epoch}: Metrics: {new_metrics}")

            # 更新最佳性能指标
            if current_metric_value is not None and current_metric_value > best_metric_value:
                best_metric_value = current_metric_value
                best_epoch = epoch
                save_model(epoch, args, metrics, model_without_ddp, optimizer, mode='best')

            cls_mf1 = metrics['分类指标'][-1] if metrics['分类指标'] else None
            print(metrics)

            # 保存最近一次的模型状态
            save_model(epoch, args, metrics, model_without_ddp, optimizer, mode='recent')

    if args.distributed:
        cleanup()



def train_one_epoch(model, train_loader, optimizer, max_norm, epoch, criterion, rank):
    model.train()
    if args.distributed:
        train_loader.sampler.set_epoch(epoch)

    iterator = train_loader
    if rank == 0:
        time_string = time.strftime('[%D-%H:%M:%S]', time.localtime())
        iterator = tqdm(train_loader, file=sys.stdout)
        iterator.set_description(f"{time_string} Train epoch-{epoch}")

    reg_tl = cls_tl = 0
    for data_iter_step, (images, points, labels, lengths) in enumerate(iterator):
        # warmup lr
        adjust_learning_rate(args,
                             optimizer,
                             data_iter_step / len(iterator) + epoch,
                             rank)

        images = images.cuda(rank)
        points = points.cuda(rank)
        labels = labels.cuda(rank)

        # all points N×2
        targets = {'gt_nums': lengths,
                   'gt_points': [points_seq[points_seq != -1].reshape(-1, 2) for points_seq in points],
                   'gt_labels': [label_seq[label_seq != -1] for label_seq in labels]}

        cell_ratios = [[(targets['gt_labels'][i] == c).sum().item() for c in range(4)] for i in range(images.size(0))]
        cell_ratios = torch.tensor(cell_ratios, dtype=torch.float).cuda(rank)
        cell_ratios = cell_ratios / cell_ratios.sum(1).unsqueeze(-1)
        targets['cell_ratios'] = cell_ratios
        # print("epoch:", epoch)
        outputs = model(images)
        losses = criterion(outputs, targets)
        loss = losses.sum()
        loss.backward()

        # from collections import OrderedDict
        # grads = OrderedDict()
        # for name, params in model.named_parameters():
        #     grad = params.grad
        #     if grad is not None:
        #         grads[name] = grad.norm().item()

        if max_norm > 0:  # clip gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        optimizer.zero_grad()
        losses /= args.world_size

        if args.distributed:
            dist.all_reduce(losses, op=dist.ReduceOp.SUM)
        reg_tl += losses[0].item()
        cls_tl += losses[1].item()

        # print("rank:", rank, "  reg_tl:", reg_tl, "  cls_tl:", cls_tl)

    print(f'回归损失: {reg_tl}, 分类损失: {cls_tl}')


class Evaluator:
    def __init__(self, data_loader_val, num_classes):
        self.data_loader = data_loader_val
        self.num_classes = num_classes
        self.gds = []
        for index, sample in enumerate(data_loader_val.dataset.data):
            data = sample
            files = data_loader_val.dataset.files[index]
            sample = data_loader_val.dataset.read_data(data, files)
            self.gds.append(tuple(sample.values())[1:])

    @torch.no_grad()
    def predict_scores(self, model, images, apply_deduplication=False):
        h, w = images.shape[-2:]
        outputs = model(images)

        points = outputs['pnt_coords'][0].cpu().numpy()
        scores = torch.softmax(outputs['cls_logits'][0], dim=-1).cpu().numpy()

        cross_border_index = (points[:, 0] < 0) | (points[:, 0] >= w) | (points[:, 1] < 0) | (points[:, 1] >= h)
        points = points[~cross_border_index]
        scores = scores[~cross_border_index]

        classes = np.argmax(scores, axis=-1)
        reserved_index = classes < self.num_classes

        if apply_deduplication:
            pd_points, pd_classes, pred_scores = self.deduplicate(points[reserved_index], scores[reserved_index])
        else:
            pd_points = points[reserved_index]
            pd_classes = classes[reserved_index]
            pred_scores = scores[reserved_index]

        return pd_points, pd_classes, pred_scores

    def draw_roc(self, model, effective_matching_dis=12, rank=0):
        eps = 1e-8
        model.eval()

        cls_pn, cls_tn, cls_rn = list(torch.zeros(self.num_classes).cuda(rank) for _ in range(3))
        det_rn, det_tn, det_pn = list(torch.zeros(1).cuda(rank) for _ in range(3))
        for i, (images, points, labels, lengths) in enumerate(self.data_loader):

            if i % args.world_size != rank:
                continue

            images = images.cuda(non_blocking=True)
            pd_points, pd_classes, pd_scores = self.predict_scores(model, images, apply_deduplication=True)
            gd_points = np.zeros((0, 2), dtype=int)
            for c in range(self.num_classes):
                category_pd_points = pd_points[pd_classes == c]
                category_gd_points = self.gds[i][c]
                # print(pd_points)
                # print(category_pd_points,category_gd_points)
                # print(gd_points.shape,category_gd_points.shape)
                gd_points = np.concatenate([gd_points, category_gd_points], axis=0)

                pred_num, gd_num = len(category_pd_points), len(category_gd_points)

                cls_pn[c] += pred_num
                cls_tn[c] += gd_num

                if pred_num and gd_num:
                    right_num, _ = binary_match(category_pd_points, category_gd_points,
                                                threshold_distance=effective_matching_dis)
                    # print('right num 1: {}'.format(right_num))
                    cls_rn[c] += right_num

            det_pn += len(pd_points)
            det_tn += len(gd_points)

            if len(pd_points) and len(gd_points):
                # right_num, _ = binary_match(pd_points, gd_points, threshold_distance=effective_matching_dis)
                right_num, match_pred_index, match_gd_index = binary_match_predAndgd(pd_points, gd_points,
                                                                                     threshold_distance=effective_matching_dis)
                matched_pd_points = pd_points[match_pred_index]
                # print('right num 2: {}'.format(right_num))

                det_rn += right_num

        if args.distributed:
            dist.all_reduce(det_rn, op=dist.ReduceOp.SUM)
            dist.all_reduce(det_tn, op=dist.ReduceOp.SUM)
            dist.all_reduce(det_pn, op=dist.ReduceOp.SUM)

            dist.all_reduce(cls_pn, op=dist.ReduceOp.SUM)
            dist.all_reduce(cls_tn, op=dist.ReduceOp.SUM)
            dist.all_reduce(cls_rn, op=dist.ReduceOp.SUM)

        det_r = det_rn / (det_tn + eps)
        det_p = det_rn / (det_pn + eps)
        det_f1 = (2 * det_r * det_p) / (det_p + det_r + eps)

        cls_r = cls_rn / (cls_tn + eps)
        cls_p = cls_rn / (cls_pn + eps)
        cls_f1 = (2 * cls_r * cls_p) / (cls_r + cls_p + eps)

        metrics = {'检测指标': [det_p.item(), det_r.item(), det_f1.item()],
                   '分类精度': cls_p.tolist(), '分类召回': cls_r.tolist(), '分类F1': cls_f1.tolist(),
                   '分类指标': [cls_p.mean().item(), cls_r.mean().item(), cls_f1.mean().item()]}
        return metrics

    @torch.no_grad()
    def predict(self, model, images, apply_deduplication: bool = False):
        h, w = images.shape[-2:]

        outputs = model(images)

        points = outputs['pnt_coords'][0].cpu().numpy()
        scores = torch.softmax(outputs['cls_logits'][0], dim=-1).cpu().numpy()
        print(points)
        print(scores)

        cross_border_index = (points[:, 0] < 0) | (points[:, 0] >= w) | (points[:, 1] < 0) | (points[:, 1] >= h)
        points = points[~cross_border_index]
        scores = scores[~cross_border_index]
        print(points)
        print(scores)

        classes = np.argmax(scores, axis=-1)
        reserved_index = classes < args.num_classes

        if apply_deduplication:
            print(deduplicate(points[reserved_index], scores[reserved_index], 15))
            return deduplicate(points[reserved_index], scores[reserved_index], 15)
        else:
            print(points[reserved_index], classes[reserved_index])
            return points[reserved_index], classes[reserved_index]

    def deduplicate(self, points, scores, threshold=15):
        """去除重复的预测点并返回对应的类别"""
        if len(points) == 0:
            return points, [], scores

        # 使用KDTree来查找附近的点
        tree = S.KDTree(points)
        unique_indices = set(range(len(points)))

        for i, point in enumerate(points):
            if i not in unique_indices:
                continue

            # 查找在threshold范围内的所有点
            indices = tree.query_ball_point(point, r=threshold)
            if len(indices) > 1:
                # 保留得分最高的点
                best_idx = max(indices, key=lambda idx: scores[idx][0])
                unique_indices.intersection_update(set([best_idx]))

        unique_indices = list(unique_indices)
        unique_points = points[unique_indices]
        unique_scores = scores[unique_indices]
        unique_classes = np.argmax(unique_scores, axis=-1)  # 假设scores是softmax后的概率分布

        return unique_points, unique_classes, unique_scores

    import torch.distributed as dist

    def calculate_metrics(self, model, effective_matching_dis=12, rank=0):
        print(f"Rank {rank}: 开始计算指标...")
        eps = 1e-8
        model.eval()

        cls_pn = torch.zeros(self.num_classes).cuda(rank)
        cls_tn = torch.zeros(self.num_classes).cuda(rank)
        cls_rn = torch.zeros(self.num_classes).cuda(rank)

        det_pn = torch.zeros(1).cuda(rank)
        det_tn = torch.zeros(1).cuda(rank)
        det_rn = torch.zeros(1).cuda(rank)

        with torch.no_grad():
            for i, (images, points, labels, lengths) in enumerate(self.data_loader):

                    if i % args.world_size != rank:
                        continue
                    print(f"Rank {rank}: 处理批次 {i}")
                    images = images.cuda(non_blocking=True)
                    pd_points, pd_classes, pred_scores = self.predict_scores(model, images, apply_deduplication=True)
                    gd_points = np.zeros((0, 2), dtype=int)

                    for c in range(self.num_classes):
                        category_pd_points = pd_points[pd_classes == c]
                        category_gd_points = self.gds[i][c]

                        # Check if there are any predictions for this class
                        if len(category_pd_points) == 0 or len(category_gd_points) == 0:
                            continue

                        category_pred_scores = pred_scores[pd_classes == c][:, 0]

                        gd_points = np.concatenate([gd_points, category_gd_points], axis=0)

                        pred_num, gd_num = len(category_pd_points), len(category_gd_points)

                        cls_pn[c] += pred_num
                        cls_tn[c] += gd_num

                        if pred_num and gd_num:
                            right_num = self.get_tp(category_pd_points, category_pred_scores, category_gd_points,
                                                    thr=effective_matching_dis)
                            cls_rn[c] += right_num

                    det_pn += len(pd_points)
                    det_tn += len(gd_points)

                    if len(pd_points) and len(gd_points):
                        pred_scores_all = np.sum(pred_scores[:, :-1], axis=1) if pred_scores.size else np.array([])
                        right_num = self.get_tp(pd_points, pred_scores_all, gd_points, thr=effective_matching_dis)
                        det_rn += right_num


        if hasattr(args, 'distributed') and args.distributed:
            print(f"Rank {rank}: 在进行 all_reduce 前")
            dist.barrier()  # 确保所有进程在此同步
            print(f"Rank {rank}: 在屏障后，准备进行 all_reduce")

            dist.all_reduce(det_rn, op=dist.ReduceOp.SUM)
            dist.all_reduce(det_tn, op=dist.ReduceOp.SUM)
            dist.all_reduce(det_pn, op=dist.ReduceOp.SUM)
            print(det_rn)
            dist.all_reduce(cls_pn, op=dist.ReduceOp.SUM)
            dist.all_reduce(cls_tn, op=dist.ReduceOp.SUM)
            dist.all_reduce(cls_rn, op=dist.ReduceOp.SUM)

            print(f"Rank {rank}: 完成 all_reduce")
        print(det_rn)
        det_precision = det_rn / (det_pn + eps)
        print("计算")
        det_recall = det_rn / (det_tn + eps)
        det_f1 = (2 * det_precision * det_recall) / (det_precision + det_recall + eps)

        cls_precision = cls_rn / (cls_pn + eps)
        cls_recall = cls_rn / (cls_tn + eps)
        cls_f1 = (2 * cls_precision * cls_recall) / (cls_precision + cls_recall + eps)

        metrics = {
            'Detection': {
                'Precision': det_precision.item(),
                'Recall': det_recall.item(),
                'F1 Score': det_f1.item()
            },
            'Classification': {
                'Precision': cls_precision.tolist(),
                'Recall': cls_recall.tolist(),
                'F1 Score': cls_f1.tolist(),
                'Mean Precision': cls_precision.mean().item() if cls_precision.numel() > 0 else 0.0,
                'Mean Recall': cls_recall.mean().item() if cls_recall.numel() > 0 else 0.0,
                'Mean F1 Score': cls_f1.mean().item() if cls_f1.numel() > 0 else 0.0
            }
        }

        print(f"计算出的指标: {metrics}")  # 调试输出

        return metrics

    # def calculate_metrics(self, model, effective_matching_dis=12, rank=0):
    #     eps = 1e-8
    #     model.eval()
    #
    #     cls_pn = torch.zeros(self.num_classes).cuda(rank)
    #     cls_tn = torch.zeros(self.num_classes).cuda(rank)
    #     cls_rn = torch.zeros(self.num_classes).cuda(rank)
    #
    #     det_pn = torch.zeros(1).cuda(rank)
    #     det_tn = torch.zeros(1).cuda(rank)
    #     det_rn = torch.zeros(1).cuda(rank)
    #
    #     with torch.no_grad():
    #         for i, (images, points, labels, lengths) in enumerate(self.data_loader):
    #
    #             if i % args.world_size != rank:
    #                 continue
    #
    #             images = images.cuda(non_blocking=True)
    #             pd_points, pd_classes, pred_scores = self.predict_scores(model, images, apply_deduplication=True)
    #             gd_points = np.zeros((0, 2), dtype=int)
    #
    #             for c in range(self.num_classes):
    #                 category_pd_points = pd_points[pd_classes == c]
    #                 category_gd_points = self.gds[i][c]
    #
    #                 # Check if there are any predictions for this class
    #                 if len(category_pd_points) == 0 or len(category_gd_points) == 0:
    #                     continue
    #
    #                 category_pred_scores = pred_scores[pd_classes == c][:, 0]
    #
    #                 gd_points = np.concatenate([gd_points, category_gd_points], axis=0)
    #
    #                 pred_num, gd_num = len(category_pd_points), len(category_gd_points)
    #
    #                 cls_pn[c] += pred_num
    #                 cls_tn[c] += gd_num
    #
    #                 if pred_num and gd_num:
    #                     right_num = self.get_tp(category_pd_points, category_pred_scores, category_gd_points,
    #                                             thr=effective_matching_dis)
    #                     cls_rn[c] += right_num
    #
    #             det_pn += len(pd_points)
    #             det_tn += len(gd_points)
    #
    #             if len(pd_points) and len(gd_points):
    #                 pred_scores_all = np.sum(pred_scores[:, :-1], axis=1) if pred_scores.size else np.array([])
    #                 right_num = self.get_tp(pd_points, pred_scores_all, gd_points, thr=effective_matching_dis)
    #                 det_rn += right_num
    #
    #     if args.distributed:
    #         dist.all_reduce(det_rn, op=dist.ReduceOp.SUM)
    #         dist.all_reduce(det_tn, op=dist.ReduceOp.SUM)
    #         dist.all_reduce(det_pn, op=dist.ReduceOp.SUM)
    #
    #         dist.all_reduce(cls_pn, op=dist.ReduceOp.SUM)
    #         dist.all_reduce(cls_tn, op=dist.ReduceOp.SUM)
    #         dist.all_reduce(cls_rn, op=dist.ReduceOp.SUM)
    #
    #     det_precision = det_rn / (det_pn + eps)
    #     det_recall = det_rn / (det_tn + eps)
    #     det_f1 = (2 * det_precision * det_recall) / (det_precision + det_recall + eps)
    #
    #     cls_precision = cls_rn / (cls_pn + eps)
    #     cls_recall = cls_rn / (cls_tn + eps)
    #     cls_f1 = (2 * cls_precision * cls_recall) / (cls_precision + cls_recall + eps)
    #
    #     metrics = {
    #         'Detection': {
    #             'Precision': det_precision.item(),
    #             'Recall': det_recall.item(),
    #             'F1 Score': det_f1.item()
    #         },
    #         'Classification': {
    #             'Precision': cls_precision.tolist(),
    #             'Recall': cls_recall.tolist(),
    #             'F1 Score': cls_f1.tolist(),
    #             'Mean Precision': cls_precision.mean().item() if cls_precision.numel() > 0 else 0.0,
    #             'Mean Recall': cls_recall.mean().item() if cls_recall.numel() > 0 else 0.0,
    #             'Mean F1 Score': cls_f1.mean().item() if cls_f1.numel() > 0 else 0.0
    #         }
    #     }
    #
    #     print(f"Calculated metrics: {metrics}")  # Debug output
    #
    #     return metrics



    def get_tp(self, pred_points, pred_scores, gd_points, thr=15):
        sorted_pred_indices = np.argsort(-pred_scores)
        sorted_pred_points = pred_points[sorted_pred_indices]

        unmatched = np.ones(len(gd_points), dtype=bool)
        dis = S.distance_matrix(sorted_pred_points, gd_points)

        for i in range(len(pred_points)):
            if not np.any(unmatched):
                break
            min_index = dis[i, unmatched].argmin()
            if dis[i, unmatched][min_index] <= thr:
                unmatched[np.where(unmatched)[0][min_index]] = False

        return sum(~unmatched)

    def visual_analysis(self, model, output_dir='vis_results/normal'):
        from skimage import io

        model.eval()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        cell_classes = ['浆细胞', '淋巴细胞', '嗜酸性粒细胞', '中性粒细胞']  # 0,1,2,3
        colors = [(234, 255, 0),  (0, 238, 255), (255, 0, 0), (102, 255, 0)]  # BXR

        # colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)]  # PDL1
        # colors = [(102, 255, 0), (255, 0, 0), (234, 255, 0), (0, 238, 255)]  # BXR
        # colors = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255), (255, 204, 153), (255, 153, 255), # PDL1_all
        #           (204, 0, 255), (153, 255, 204), (0, 255, 255), (204, 153, 0)]

        # colors = [(254, 57, 151), (255, 102, 59), (23, 26, 29), (253, 9, 55), (65, 84, 174), (6, 54, 24)]  # HER2
        # colors = [(255, 192, 203), (255, 102, 51), (0, 255, 0), (255, 0, 51), (64, 81, 181), (102, 0, 102)]  # HER2
        # color_dict = {
        #     "0": (255, 192, 203),
        #     "1": (255, 102, 51),
        #     "2": (0, 255, 0),
        #     "3": (255, 0, 51),
        #     "4": (64, 81, 181),
        #     "5": (102, 0, 102),
        # }
        # label_dict = {
        #     '微弱的不完整膜阳性肿瘤细胞': 0,
        #     '弱-中等的完整细胞膜阳性肿瘤细胞': 1,
        #     '阴性肿瘤细胞': 2,
        #     '强度的完整细胞膜阳性肿瘤细胞': 3,
        #     '中-强度的不完整细胞膜阳性肿瘤细胞': 4,
        #     '纤维细胞': 5,
        #     '淋巴细胞': 5,
        #     '组织细胞': 5,
        # }

        # cell_classes = ['阴性细胞', '阳性细胞']  # 0,1
        # colors = [(0, 238, 255), (255, 0, 0), ]  # yingguang

        # cell_classes =['阳性细胞', '阴性细胞', '不确定细胞', ]   # 胸腹水 0,1,2
        # colors = [(255, 0, 0),  (0, 255, 0), (255, 255, 51), (102, 255, 0), ]  # 胸腹水

        # cell_classes = ['合体结节']
        # colors = [(0, 255, 255)]
        # cell_classes = ['含铁血黄素']   # 胎膜 0， 1， 2
        # colors = [(255, 255, 103)]

        for i, (images, points, labels, lengths) in enumerate(self.data_loader):
            img_name = os.path.basename(self.data_loader.dataset.files[i])
            print(f'Visualizing --- {img_name}')

            images = images.cuda(non_blocking=True)

            pd_points, pd_classes = self.predict(model, images, apply_deduplication=True)

            # create db file
            from np2db import output2db
            template_db = r'/home/zhangwanqi/code/code_cell/slice.db'
            db_save_root = "/home/zhangwanqi/code/code_cell/employ/"
            table_name = "None"
            save_folder = os.path.join(db_save_root, img_name)
            # label_name = {0: "组织细胞", 1: "淋巴细胞", 2: "中性粒细胞"}
            # label_markgroup = {0: 520, 1: 519, 2: 518}
            # label_name = {0: "含铁血黄素"}
            # label_markgroup = {0: 520}
            # label_name = {0: "合体结节"}
            # label_markgroup = {0: 520}
            label_name = {0: "浆细胞", 1: "淋巴细胞", 2: "嗜酸粒细胞", 3: "中性粒细胞"}
            label_markgroup = {0: 520, 1: 519, 2: 518, 3: 517}
            output2db(pd_points, pd_classes, template_db, save_folder, label_markgroup, table_name)

            # draw pred points
            # image = self.data_loader.dataset.data[i]['image'].copy()
            image = \
            self.data_loader.dataset.read_data(self.data_loader.dataset.data[i], self.data_loader.dataset.files[i])[
                'image'].copy()
            image1 = image.copy()
            io.imsave(f"{output_dir}/{img_name}_ori.jpg", image1, check_contrast=False)
            for c, (x, y) in zip(pd_classes.astype(int), pd_points.astype(int)):
                cv.line(image, (x - 6, y), (x + 6, y), color=colors[c], thickness=4, lineType=cv.LINE_AA)
                cv.line(image, (x, y - 6), (x, y + 6), color=colors[c], thickness=4, lineType=cv.LINE_AA)
            print(zip(pd_classes.astype(int), pd_points.astype(int)))

            io.imsave(f"{output_dir}/{img_name}_pred.jpg", image, check_contrast=False)

            # draw gd points
            for c in range(self.num_classes):
                for (x, y) in self.gds[i][c].astype(int):
                    cv.circle(image, (x, y), radius=12, color=colors[c], thickness=1, lineType=cv.LINE_AA)
                    cv.line(image1, (x - 6, y), (x + 6, y), color=colors[c], thickness=4, lineType=cv.LINE_AA)
                    cv.line(image1, (x, y - 6), (x, y + 6), color=colors[c], thickness=4, lineType=cv.LINE_AA)
            io.imsave(f"{output_dir}/{img_name}_ori_mark.jpg", image1, check_contrast=False)

            io.imsave(f"{output_dir}/{img_name}", image, check_contrast=False)

    @torch.no_grad()
    def match_procedure(self, model):
        model.eval()
        matcher = build_matcher(args)
        for i, (images, points, labels, lengths) in enumerate(self.data_loader):
            img_name = self.data_loader.dataset.files[i].split('.')[0]
            print(f'Processing --- ({img_name})')

            images = images.cuda(non_blocking=True)
            points = points.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            outputs = model(images)

            targets = {'gt_nums': lengths,
                       'gt_points': [points_seq[points_seq != -1].reshape(-1, 2) for points_seq in points],
                       'gt_labels': [label_seq[label_seq != -1] for label_seq in labels]}

            indices = matcher(outputs, targets)
            np.save(f'/home/zhangwanqi/code/code_cell/employ/{img_name}_indices',
                    [indices[0][0].cpu().numpy(), indices[0][1].cpu().numpy()])

            cls_scores = outputs['cls_logits'][0].softmax(-1).cpu().numpy()
            points = outputs['pnt_coords'][0].cpu().numpy()

            reg_attn = outputs['reg_attn'][0, ..., 0].cpu().numpy()
            cls_attn = outputs['cls_attn'][0, ..., 0].cpu().numpy()

            np.save(f'/home/zhangwanqi/code/code_cell/employ_0121/{img_name}_scores', cls_scores)
            np.save(f'/home/zhangwanqi/code/code_cell/employ_0121/{img_name}_points', points.astype(int))
            np.save(f'/home/zhangwanqi/code/code_cell/employ_0221/{img_name}_cls_attn', reg_attn)
            np.save(f'/home/zhangwanqi/code/code_cell/employ_0121/{img_name}_reg_attn', cls_attn)




def eval(ckpt_path):
    print("Starting evaluation...")

    # 构建模型
    model = build_model(args)
    print("Model built.")

    # 加载预训练模型的检查点
    ckpt = torch.load(ckpt_path, map_location='cpu')
    print(f"Checkpoint loaded from {ckpt_path}. Epoch: {ckpt['epoch']}, Metrics: {ckpt['metrics']}")

    # 构建测试集的数据集对象
    dataset_val = build_dataset(args, 'test')
    print("Dataset for validation/test built.")

    # 创建数据加载器
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_pad
    )
    print("DataLoader created.")

    # 测试数据加载器是否可以正常工作
    try:
        for i, batch in enumerate(data_loader_val):
            if i == 0:
                print(f"First batch loaded successfully: {batch}")
                break
    except Exception as e:
        print(f"Error while loading batches from DataLoader: {e}")
        return

    # 更新模型状态字典
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in ckpt['model'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print("Model state dict updated with pre-trained weights.")

    # 将模型移动到GPU上
    model.cuda()
    print("Model moved to GPU.")

    # 创建评估器对象，并传入正确的参数
    evaluator = Evaluator(data_loader_val, num_classes=4)
    print("Evaluator initialized.")

    # 计算模型在数据集上的评估指标
    metrics = evaluator.calculate_metrics(model, effective_matching_dis=args.match_dis)
    print("Evaluation Metrics:", metrics)


# evaluator.visual_analysis(model, output_dir='vis_results/beiertong_20231022')
    # evaluator.match_procedure(model)


if __name__ == '__main__':

    parser = get_args_parser()
    args = parser.parse_args()

    # os.environ['MASTER_PORT'] = '29600'

    init_distributed_mode(args)

    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # train()
    eval('/home/zhangwanqi/code/code_cell/pth/bjrt_cell_20250217_resnet50_eos_coef_0_5_recent.pth')

