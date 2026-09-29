import os
import sys
import argparse
import csv
import json
import math
import random
#import wandb
import cv2 as cv
import numpy as np
import time

from utils import *
from tqdm import tqdm

# from dataset import build_dataset
from dataset_zy_src import build_dataset
from models.detr import build_model
from loss import build_criterion
from matcher import build_matcher
from lr_sched import adjust_learning_rate
from prototype_e1_checkpoint import (
    load_baseline_checkpoint,
    save_initialization_report,
)

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
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--start_eval', default=100, type=int)
    parser.add_argument('--use_proto_e1', action='store_true')
    parser.add_argument('--proto_hidden_dim', default=128, type=int)
    parser.add_argument('--proto_embed_dim', default=64, type=int)
    parser.add_argument('--proto_max_pos', default=32, type=int)
    parser.add_argument('--proto_max_neg', default=64, type=int)
    parser.add_argument('--proto_pos_radius', default=15.0, type=float)
    parser.add_argument('--proto_neg_radius', default=30.0, type=float)
    parser.add_argument('--proto_margin', default=0.1, type=float)
    parser.add_argument('--proto_beta', default=16.0, type=float)
    parser.add_argument('--proto_loss_coef', default=0.05, type=float)
    parser.add_argument('--proto_grad_audit_interval', default=500, type=int)
    parser.add_argument(
        '--debug_max_train_batches', default=0, type=int,
        help='Stop each epoch after this many batches; 0 disables the limit',
    )

    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument(
        '--init_checkpoint', default='', type=str,
        help='Initialize model weights only; optimizer and epoch start fresh',
    )
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--num_classes', type=int, default=1,
                        help="Number of cell categories")

    # * Loss
    parser.add_argument('--reg_loss_coef', default=2e-3, type=float)
    parser.add_argument('--cls_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.3, type=float,
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
    parser.add_argument('--num_workers', default=4, type=int)

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


def _append_csv_row(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def train():
    if args.distributed:
        rank = args.gpu
    else:
        rank = 0
    # 创建输出目录（如果不存在）
    # 创建输出目录（如果不存在）
    if rank == 0 and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        log_file = os.path.join(args.output_dir, 'training_log.txt')

        # === 核心修改：判断是否为续训，动态决定写入模式 ===
        write_mode = 'a' if args.resume else 'w'
        with open(log_file, write_mode) as f:
            if not args.resume:
                f.write("Training Log\n")
            else:
                f.write(f"\n\n{'='*40}\n")
                f.write(f"--- 触发断点续训: 从 {args.resume} 恢复 ---\n")
                f.write(f"{'='*40}\n")
        # #run = wandb.init(project='wandb_usage', entity="zhouzh")
        # run.name = run.id
        # run.save()
        # cfg = wandb.config
        for k, v in args.__dict__.items():
            # setattr(cfg, k, v)
            pass

    if args.resume and args.init_checkpoint:
        raise ValueError('--resume and --init_checkpoint are mutually exclusive')

    model = build_model(args).cuda(rank)
    model_without_ddp = model

    initialization_report = None
    if args.init_checkpoint:
        initialization_report = load_baseline_checkpoint(
            model_without_ddp,
            args.init_checkpoint,
        )
        if rank == 0:
            print(
                '[Checkpoint-init] '
                f"path={initialization_report['checkpoint']}, "
                f"loaded_tensors={initialization_report['loaded_tensors']}, "
                f"new_adapter_tensors={len(initialization_report['missing_new_parameters'])}"
            )

    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)
        model_without_ddp = model.module

    matcher = build_matcher(args)
    proto_adapter = (
        model_without_ddp.proto_metric_adapter
        if bool(getattr(args, 'use_proto_e1', False)) else None
    )
    criterion = build_criterion(rank, matcher, args, proto_adapter=proto_adapter)
    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    data_loaders = construct_dataset()
    evaluator = Evaluator(data_loaders['test'])
    first_eval = True

    if rank == 0 and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, 'config.json'), 'w') as handle:
            json.dump(vars(args), handle, indent=2, sort_keys=True)
        if initialization_report is not None:
            report_path = save_initialization_report(
                initialization_report, args.output_dir
            )
            print(f'[Checkpoint-init] report={report_path}')

    # load checkpoint
    metrics = load_checkpoint(args, model_without_ddp, optimizer) if args.resume else {}
    max_cls_mf1 = metrics.get('分类F1', 0)

    # 初始化损失记录列表
    reg_losses =[]
    cls_losses = []
    proto_losses = []
    total_losses =[]

    for epoch in range(args.start_epoch, args.epochs):
        # train_one_epoch(model, data_loaders['train'], optimizer, args.clip_max_norm, epoch, criterion, rank)

        # 训练一个epoch并获取损失
        reg_tl, cls_tl, proto_tl, proto_diag = train_one_epoch(
            model, data_loaders['train'], optimizer, args.clip_max_norm,
            epoch, criterion, rank
        )

        # 记录损失
        reg_losses.append(reg_tl)
        cls_losses.append(cls_tl)
        proto_losses.append(proto_tl)
        total_losses.append(reg_tl + cls_tl + proto_tl)


        # 写入训练损失到文件
        if rank == 0 and args.output_dir:
            log_file = os.path.join(args.output_dir, 'training_log.txt')
            with open(log_file, 'a') as f:
                f.write(f"Epoch {epoch}: Regression Loss={reg_tl:.4f}, Classification Loss={cls_tl:.4f}, Weighted Prototype Loss={proto_tl:.6f}, Total Loss={reg_tl + cls_tl + proto_tl:.4f}\n")
            if bool(getattr(args, 'use_proto_e1', False)):
                row = {'epoch': epoch, 'weighted_loss_proto': proto_tl}
                row.update(proto_diag)
                _append_csv_row(os.path.join(args.output_dir, 'proto_diagnostics.csv'), row)

        if trigger_eval(epoch, args.start_eval):
            if first_eval:
                save_model(epoch, args, metrics, model_without_ddp, optimizer, mode='best')
                first_eval = False

            metrics = evaluator.calculate_metrics(model, rank=rank, effective_matching_dis=args.match_dis)
            cls_mf1 = metrics['分类指标'][-1]
            print(metrics)

            # save most recent epoch model
            save_model(epoch, args, metrics, model_without_ddp, optimizer, mode='recent')

            # cls_mf1 = sum(metrics['分类F1'][:2]) / 2
            # print(metrics, cls_mf1)

            # 写入评估指标到文件
            if rank == 0 and args.output_dir:
                log_file = os.path.join(args.output_dir, 'training_log.txt')
                with open(log_file, 'a') as f:
                    f.write(f"Epoch {epoch} Evaluation:\n")
                    f.write(f"  Detection Precision={metrics['检测指标'][0]:.4f}, Recall={metrics['检测指标'][1]:.4f}, F1={metrics['检测指标'][2]:.4f}\n")
                    f.write(f"  Classification Precision={metrics['分类指标'][0]:.4f}, Recall={metrics['分类指标'][1]:.4f}, F1={metrics['分类指标'][2]:.4f}\n")
                    f.write(f"  Best Classification F1: {max_cls_mf1:.4f}\n")

            if rank == 0:
                # wandb.log(dict(zip(["检测精度", "检测召回", "检测F1"], metrics['检测指标'])))
                # wandb.log(dict(zip(["分类精度", "分类召回", "分类F1"], metrics['分类指标'])))
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 5))
                plt.plot(reg_losses, label='Regression Loss')
                plt.plot(cls_losses, label='Classification Loss')
                if bool(getattr(args, 'use_proto_e1', False)):
                    plt.plot(proto_losses, label='Weighted Prototype Loss')
                plt.plot(total_losses, label='Total Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training Loss Curve')
                plt.legend()

                # 保存到输出目录
                loss_curve_path = os.path.join(args.output_dir, 'loss_curve.png')
                plt.savefig(loss_curve_path)
                print(f"Loss curve saved to {loss_curve_path}")

                if max_cls_mf1 < cls_mf1:
                    max_cls_mf1 = cls_mf1
                    if args.output_dir:
                        save_model(epoch, args, metrics, model_without_ddp, optimizer, mode='best')
                print("max cls_mf1:", max_cls_mf1)

        # 新增代码：从第30个epoch开始，每隔20个epoch保存一次模型
        if epoch >= 30 and (epoch - 30) % 20 == 0 and rank == 0:
            current_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
            model_name = f"model_epoch_{epoch}_{current_time}.pth"
            model_path = os.path.join(args.output_dir, model_name)
            torch.save({
        'epoch': epoch,
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'metrics': metrics,
    }, model_path)
            print(f"Model saved to {model_path}")
            if args.output_dir:
                log_file = os.path.join(args.output_dir, 'training_log.txt')
                with open(log_file, 'a') as f:
                    f.write(f"Model saved: {model_path}\n")

    if rank == 0 and args.output_dir and bool(getattr(args, 'use_proto_e1', False)):
        final_epoch = args.epochs - 1
        final_path = os.path.join(args.output_dir, 'final_model.pth')
        torch.save(
            {
                'epoch': final_epoch,
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'metrics': metrics,
            },
            final_path,
        )
        print(f'[E1-final-checkpoint] epoch={final_epoch}, path={final_path}')

    if args.distributed:
        cleanup()


def test():
    # 统一设置根目录变量，方便后续修改
    base_dir = 'pth/0326_bd'

    data_loaders = construct_dataset()
    model = build_model(args).cuda()

    # 模型加载路径
    model_path = os.path.join(base_dir, 'best_model.pth')
    # 修改点：使用 strict=False 忽略多余的键（如 proto_enhance.*）
    model.load_state_dict(torch.load(model_path)['model'], strict=False)
    model.eval()

    evaluator = Evaluator(data_loaders['test'])

    # 1. 可视化图片输出路径：pth/0326_bd/vis
    vis_dir = os.path.join(base_dir, 'vis')
    evaluator.visual_analysis(model, output_dir=vis_dir)

    # 计算测试指标
    metrics = evaluator.calculate_metrics(model, effective_matching_dis=args.match_dis)
    print("测试指标:")
    print(f"检测指标: 精度={metrics['检测指标'][0]:.4f}, 召回={metrics['检测指标'][1]:.4f}, F1={metrics['检测指标'][2]:.4f}")
    print(f"分类指标: 精度={metrics['分类指标'][0]:.4f}, 召回={metrics['分类指标'][1]:.4f}, F1={metrics['分类指标'][2]:.4f}")
    print(f"MSE: {metrics['MSE']:.4f}, MAE: {metrics['MAE']:.4f}")

    # 处理文件路径不存在的情况（写txt操作）
    # import os
    file_path = os.path.join(base_dir, 'test_results.txt')
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)   # 如果 base_dir 不存在也会自动创建

    with open(file_path, 'w') as f:
        f.write("=== 测试结果 ===\n")
        f.write(f"检测指标: {metrics['检测指标']}\n")
        f.write(f"分类指标: {metrics['分类指标']}\n")
        f.write(f"MSE: {metrics['MSE']}\n")
        f.write(f"MAE: {metrics['MAE']}\n")


def train_one_epoch(model, train_loader, optimizer, max_norm, epoch, criterion, rank):
    model.train()
    if args.distributed:
        train_loader.sampler.set_epoch(epoch)

    iterator = train_loader
    if rank == 0:
        time_string = time.strftime('[%D-%H:%M:%S]', time.localtime())
        iterator = tqdm(train_loader, file=sys.stdout)
        iterator.set_description(f"{time_string} Train epoch-{epoch}")

    reg_tl = cls_tl = proto_tl = 0.0
    proto_metric_keys = (
        'loss_proto_raw', 'num_positive', 'num_negative', 'proxy_cosine',
        'pos_margin_mean', 'pos_margin_std', 'neg_margin_mean',
        'neg_margin_std', 'proto_pos_acc', 'proto_neg_acc',
    )
    proto_metric_sums = {key: 0.0 for key in proto_metric_keys}
    proto_metric_counts = {key: 0.0 for key in proto_metric_keys}
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

        if bool(getattr(args, 'use_proto_e1', False)):
            interval = int(getattr(args, 'proto_grad_audit_interval', 500))
            if interval > 0 and (
                data_iter_step == 0 or (data_iter_step + 1) % interval == 0
            ):
                cls_features = outputs['cls_features']
                det_grad = torch.autograd.grad(
                    losses[0] + losses[1],
                    cls_features,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                proto_grad = torch.autograd.grad(
                    losses[2],
                    cls_features,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                if det_grad is None:
                    det_grad = torch.zeros_like(cls_features)
                if proto_grad is None:
                    proto_grad = torch.zeros_like(cls_features)
                det_flat = det_grad.detach().float().reshape(-1)
                proto_flat = proto_grad.detach().float().reshape(-1)
                det_norm = torch.linalg.vector_norm(det_flat)
                proto_norm = torch.linalg.vector_norm(proto_flat)
                ratio = proto_norm / (det_norm + 1e-12)
                cosine = torch.sum(det_flat * proto_flat) / (
                    det_norm * proto_norm + 1e-12
                )
                if rank == 0 and args.output_dir:
                    _append_csv_row(
                        os.path.join(args.output_dir, 'gradient_audit.csv'),
                        {
                            'epoch': epoch,
                            'iteration': data_iter_step,
                            'det_grad_norm': float(det_norm.item()),
                            'proto_grad_norm': float(proto_norm.item()),
                            'proto_to_det_grad_ratio': float(ratio.item()),
                            'grad_cosine': float(cosine.item()),
                        },
                    )

        loss.backward()

        if max_norm > 0:  # clip gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        optimizer.zero_grad()
        report_losses = losses.detach() / args.world_size

        if args.distributed:
            dist.all_reduce(report_losses, op=dist.ReduceOp.SUM)
        reg_tl += report_losses[0].item()
        cls_tl += report_losses[1].item()
        if bool(getattr(args, 'use_proto_e1', False)):
            proto_tl += report_losses[2].item()
            for key in proto_metric_keys:
                value = criterion.last_proto_metrics.get(key, float('nan'))
                if math.isfinite(float(value)):
                    proto_metric_sums[key] += float(value)
                    proto_metric_counts[key] += 1.0

        debug_limit = int(getattr(args, 'debug_max_train_batches', 0))
        if debug_limit > 0 and data_iter_step + 1 >= debug_limit:
            print(
                f'[Debug-train-limit] epoch={epoch}, '
                f'processed_batches={data_iter_step + 1}'
            )
            break

    epoch_proto_metrics = {}
    if bool(getattr(args, 'use_proto_e1', False)):
        metric_values = []
        for key in proto_metric_keys:
            metric_values.extend((proto_metric_sums[key], proto_metric_counts[key]))
        metric_tensor = torch.tensor(metric_values, dtype=torch.float64).cuda(rank)
        if args.distributed:
            dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
        metric_values = metric_tensor.cpu().tolist()
        for index, key in enumerate(proto_metric_keys):
            total = metric_values[index * 2]
            count = metric_values[index * 2 + 1]
            epoch_proto_metrics[key] = total / count if count else float('nan')
    print(
        f'回归损失: {reg_tl}, 分类损失: {cls_tl}, '
        f'加权原型损失: {proto_tl}'
    )
    # 返回本epoch的损失
    return reg_tl, cls_tl, proto_tl, epoch_proto_metrics


class Evaluator:
    def __init__(self, data_loader_val):
        super(Evaluator, self).__init__()
        self.data_loader = data_loader_val
        self.gds =[]
        for index, sample in enumerate(data_loader_val.dataset.data):
            data = sample
            files = data_loader_val.dataset.files[index]
            sample = data_loader_val.dataset.read_data(data, files)
            self.gds.append(tuple(sample.values())[1:])
        self.num_classes = args.num_classes

    @torch.no_grad()
    def predict_scores(self, model, images, apply_deduplication: bool = False):
        h, w = images.shape[-2:]
        outputs = model(images)

        points = outputs['pnt_coords'][0].cpu().numpy()
        scores = torch.softmax(outputs['cls_logits'][0], dim=-1).cpu().numpy()

        cross_border_index = (points[:, 0] < 0) | (points[:, 0] >= w) | (points[:, 1] < 0) | (points[:, 1] >= h)
        points = points[~cross_border_index]
        scores = scores[~cross_border_index]

        # scores[:, -1] = scores[:, -1]*0.3
        classes = np.argmax(scores, axis=-1)
        reserved_index = classes < args.num_classes

        if apply_deduplication:
            return deduplicate_scores(points[reserved_index], scores[reserved_index], 15)
        else:
            return points[reserved_index], classes[reserved_index]

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
                gd_points = np.concatenate([gd_points, category_gd_points], axis=0)

                pred_num, gd_num = len(category_pd_points), len(category_gd_points)

                cls_pn[c] += pred_num
                cls_tn[c] += gd_num

                if pred_num and gd_num:
                    right_num, _ = binary_match(category_pd_points, category_gd_points,
                                                threshold_distance=effective_matching_dis)
                    cls_rn[c] += right_num

            det_pn += len(pd_points)
            det_tn += len(gd_points)

            if len(pd_points) and len(gd_points):
                right_num, match_pred_index, match_gd_index = binary_match_predAndgd(pd_points, gd_points,
                                                                                     threshold_distance=effective_matching_dis)
                matched_pd_points = pd_points[match_pred_index]
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

        metrics = {'检测指标':[det_p.item(), det_r.item(), det_f1.item()],
                   '分类精度': cls_p.tolist(), '分类召回': cls_r.tolist(), '分类F1': cls_f1.tolist(),
                   '分类指标':[cls_p.mean().item(), cls_r.mean().item(), cls_f1.mean().item()]}
        return metrics

    @torch.no_grad()
    def predict(self, model, images, apply_deduplication: bool = False):
        h, w = images.shape[-2:]
        outputs = model(images)

        points = outputs['pnt_coords'][0].cpu().numpy()
        scores = torch.softmax(outputs['cls_logits'][0], dim=-1).cpu().numpy()

        cross_border_index = (points[:, 0] < 0) | (points[:, 0] >= w) | (points[:, 1] < 0) | (points[:, 1] >= h)
        points = points[~cross_border_index]
        scores = scores[~cross_border_index]

        classes = np.argmax(scores, axis=-1)
        reserved_index = classes < args.num_classes

        if apply_deduplication:
            return deduplicate(points[reserved_index], scores[reserved_index], 15)
        else:
            return points[reserved_index], classes[reserved_index]

    def calculate_metrics(self, model, effective_matching_dis=12, rank=0):
        eps = 1e-8
        model.eval()

        cls_pn, cls_tn, cls_rn = list(torch.zeros(self.num_classes).cuda(rank) for _ in range(3))
        det_rn, det_tn, det_pn = list(torch.zeros(1).cuda(rank) for _ in range(3))
        mse_sum = 0.0
        mae_sum = 0.0
        total_matches = 0

        for i, (images, points, labels, lengths) in enumerate(self.data_loader):

            if i % args.world_size != rank:
                continue

            images = images.cuda(non_blocking=True)
            pd_points, pd_classes, pred_scores = self.predict_scores(model, images, apply_deduplication=True)
            gd_points = np.zeros((0, 2), dtype=int)
            for c in range(self.num_classes):
                category_pd_points = pd_points[pd_classes == c]
                category_gd_points = self.gds[i][c]
                category_pred_scores = pred_scores[pd_classes == c][:, 0]

                gd_points = np.concatenate([gd_points, category_gd_points], axis=0)

                pred_num, gd_num = len(category_pd_points), len(category_gd_points)

                cls_pn[c] += pred_num
                cls_tn[c] += gd_num

                if pred_num and gd_num:
                    right_num, matched_pred, matched_gd = self.get_tp(category_pd_points, category_pred_scores, category_gd_points, thr=effective_matching_dis)
                    cls_rn[c] += right_num

                    if right_num > 0:
                        squared_errors = np.sum((matched_pred - matched_gd) ** 2, axis=1)
                        absolute_errors = np.sum(np.abs(matched_pred - matched_gd), axis=1)
                        mse_sum += np.sum(squared_errors)
                        mae_sum += np.sum(absolute_errors)
                        total_matches += right_num

            det_pn += len(pd_points)
            det_tn += len(gd_points)

            if len(pd_points) and len(gd_points):
                pred_scores_global = np.sum(pred_scores[:, :-1], axis=1)
                right_num_global, _, _ = self.get_tp(pd_points, pred_scores_global, gd_points, thr=effective_matching_dis)
                det_rn += right_num_global

        # 分布式汇总
        if args.distributed:
            mse_sum_tensor = torch.tensor(mse_sum).cuda(rank)
            mae_sum_tensor = torch.tensor(mae_sum).cuda(rank)
            total_matches_tensor = torch.tensor(total_matches).cuda(rank)
            dist.all_reduce(mse_sum_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(mae_sum_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_matches_tensor, op=dist.ReduceOp.SUM)
            mse_sum = mse_sum_tensor.item()
            mae_sum = mae_sum_tensor.item()
            total_matches = total_matches_tensor.item()

        # 计算MSE和MAE
        mse = mse_sum / (total_matches + eps)
        mae = mae_sum / (total_matches + eps)

        # 原指标计算
        det_r = det_rn / (det_tn + eps)
        det_p = det_rn / (det_pn + eps)
        det_f1 = (2 * det_r * det_p) / (det_p + det_r + eps)

        cls_r = cls_rn / (cls_tn + eps)
        cls_p = cls_rn / (cls_pn + eps)
        cls_f1 = (2 * cls_r * cls_p) / (cls_r + cls_p + eps)

        metrics = {
            '检测指标':[det_p.item(), det_r.item(), det_f1.item()],
            '分类精度': cls_p.tolist(),
            '分类召回': cls_r.tolist(),
            '分类F1': cls_f1.tolist(),
            '分类指标':[cls_p.mean().item(), cls_r.mean().item(), cls_f1.mean().item()],
            'MSE': mse,
            'MAE': mae
        }
        return metrics

    def get_tp(self, pred_points, pred_scores, gd_points, thr=15):
        sorted_indices = np.argsort(-pred_scores)
        sorted_pred_points = pred_points[sorted_indices]

        unmatched = np.ones(len(gd_points), dtype=bool)
        dis = S.distance_matrix(sorted_pred_points, gd_points)

        matched_pred_indices = []
        matched_gd_indices =[]

        for i in range(len(sorted_pred_points)):
            if not np.any(unmatched):
                break
            min_index_in_unmatched = np.argmin(dis[i, unmatched])
            original_gd_index = np.where(unmatched)[0][min_index_in_unmatched]
            distance = dis[i, original_gd_index]
            if distance <= thr:
                matched_pred_indices.append(i)
                matched_gd_indices.append(original_gd_index)
                unmatched[original_gd_index] = False

        # Convert indices to original pred_points order
        original_pred_indices = sorted_indices[matched_pred_indices]
        matched_pred = pred_points[original_pred_indices]
        matched_gd = gd_points[matched_gd_indices]

        return len(matched_pred), matched_pred, matched_gd

    # 修改：完全重构了生成三种特定图像逻辑，并在 img_3 上标注 TP/FP/FN 数量
    def visual_analysis(self, model, output_dir='vis_results/normal'):
        from skimage import io
        import numpy as np

        model.eval()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        for i, (images, points, labels, lengths) in enumerate(self.data_loader):
            img_name = os.path.basename(self.data_loader.dataset.files[i])
            print(f'Visualizing --- {img_name}')

            images = images.cuda(non_blocking=True)

            # 修改点：使用 predict_scores 直接获取得分用于计算 TP/FP/FN
            pd_points, pd_classes, pred_scores = self.predict_scores(model, images, apply_deduplication=True)

            # 保持创建 db file 不变
            # from np2db import output2db
            # table_name = "None"
            # save_folder = os.path.join(db_save_root, img_name)
            # label_name = {0: "印戒细胞"}
            # label_markgroup = {0: 520}

            # output2db(pd_points, pd_classes, template_db, save_folder, label_markgroup, table_name)

            image = self.data_loader.dataset.read_data(self.data_loader.dataset.data[i],
                                                       self.data_loader.dataset.files[i])['image'].copy()

            # 初始化三种状态的图像底板
            img_1 = image.copy()  # 1-真实标注：绿色点
            img_2 = image.copy()  # 2-真实标注（绿色圈）+模型预测（红点点）
            img_3 = image.copy()  # 3-真实标注（绿色圈）+模型预测对的（红点）+假阳（蓝色十字）+漏检（紫色十字）

            # 统一颜色定义 (RGB 格式, skimage.io.imsave 需要 RGB)
            color_gt_green = (0, 255, 0)
            color_pred_red = (255, 0, 0)
            color_fp_blue = (0, 0, 255)
            color_fn_purple = (255, 0, 255)

            # 画笔粗细定义
            circle_radius = 14
            circle_thick = 3     # 圆圈厚一点
            dot_radius = 4

            # 1. 绘制真实标注 (Image 1, 2, 3通用)
            for c in range(self.num_classes):
                for (x, y) in self.gds[i][c].astype(int):
                    # 1-真实标注：绿色点
                    cv.circle(img_1, (x, y), radius=dot_radius, color=color_gt_green, thickness=-1)
                    # 2 & 3: 真实标注：绿色圈
                    cv.circle(img_2, (x, y), radius=circle_radius, color=color_gt_green, thickness=circle_thick)
                    cv.circle(img_3, (x, y), radius=circle_radius, color=color_gt_green, thickness=circle_thick)

            # 2. 绘制模型预测 (Image 2: 模型预测红点点)
            for (x, y) in pd_points.astype(int):
                cv.circle(img_2, (x, y), radius=dot_radius, color=color_pred_red, thickness=-1)

            # 3. 匹配并分析绘制 TP, FP, FN (Image 3专属)
            def get_unmatched(all_pts, matched_pts):
                if len(matched_pts) == 0: return all_pts
                if len(all_pts) == 0: return all_pts
                diff = all_pts[:, np.newaxis, :] - matched_pts[np.newaxis, :, :]
                dists = np.linalg.norm(diff, axis=2)
                min_dists = np.min(dists, axis=1)
                return all_pts[min_dists > 1e-5]

            total_tp = 0
            total_fp = 0
            total_fn = 0

            for c in range(self.num_classes):
                category_pd_points = pd_points[pd_classes == c]
                category_pred_scores = pred_scores[pd_classes == c][:, 0]
                category_gd_points = self.gds[i][c]

                if len(category_pd_points) > 0 and len(category_gd_points) > 0:
                    # 使用 get_tp 找出被正确匹配的点
                    right_num, matched_pred, matched_gd = self.get_tp(category_pd_points, category_pred_scores, category_gd_points, thr=args.match_dis)
                else:
                    matched_pred = np.array([])
                    matched_gd = np.array([])

                tp_pts = matched_pred
                fp_pts = get_unmatched(category_pd_points, matched_pred)
                fn_pts = get_unmatched(category_gd_points, matched_gd)

                total_tp += len(tp_pts)
                total_fp += len(fp_pts)
                total_fn += len(fn_pts)

                # TP 模型预测对的: 红点
                for (x, y) in tp_pts.astype(int):
                    cv.circle(img_3, (x, y), radius=dot_radius, color=color_pred_red, thickness=-1)

                # FP 假阳: 蓝色十字
                for (x, y) in fp_pts.astype(int):
                    cv.drawMarker(img_3, (x, y), color=color_fp_blue, markerType=cv.MARKER_CROSS, markerSize=12, thickness=circle_thick)

                # FN 漏检: 紫色十字
                for (x, y) in fn_pts.astype(int):
                    cv.drawMarker(img_3, (x, y), color=color_fn_purple, markerType=cv.MARKER_CROSS, markerSize=12, thickness=circle_thick)

            # 在 img_3 上标注统计信息
            text = f"TP: {total_tp}  FP: {total_fp}  FN: {total_fn}"
            cv.putText(img_3, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

            # 保存三张图片并清理后缀避免文件堆叠后缀
            img_base = os.path.splitext(img_name)[0]
            io.imsave(f"{output_dir}/{img_base}_1_gt_dots.jpg", img_1, check_contrast=False)
            io.imsave(f"{output_dir}/{img_base}_2_gt_pred.jpg", img_2, check_contrast=False)
            io.imsave(f"{output_dir}/{img_base}_3_analysis.jpg", img_3, check_contrast=False)


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
            np.save(f'./tool/result/{img_name}_indices',
                    [indices[0][0].cpu().numpy(), indices[0][1].cpu().numpy()])

            cls_scores = outputs['cls_logits'][0].softmax(-1).cpu().numpy()
            points = outputs['pnt_coords'][0].cpu().numpy()

            reg_attn = outputs['reg_attn'][0, ..., 0].cpu().numpy()
            cls_attn = outputs['cls_attn'][0, ..., 0].cpu().numpy()

            np.save(f'./tool/result/{img_name}_scores', cls_scores)
            np.save(f'./tool/result/{img_name}_points', points.astype(int))
            np.save(f'./tool/result/{img_name}_cls_attn', reg_attn)
            np.save(f'./tool/result/{img_name}_reg_attn', cls_attn)


def eval(ckpt_path):
    model = build_model(args)
    ckpt = torch.load(f'checkpoint/{ckpt_path}', map_location='cpu')
    print(ckpt['epoch'], ckpt['metrics'])
    # ckpt = torch.load(f'./{ckpt_path}', map_location='cpu')

    dataset_val = build_dataset(args, 'test')
    # dataset_val = build_dataset(args, 'train')
    data_loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                 collate_fn=collate_fn_pad)

    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in ckpt['model'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model.cuda()

    evaluator = Evaluator(data_loader_val)
    metrics = evaluator.calculate_metrics(model, effective_matching_dis=args.match_dis)

    print(metrics)

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

    train()
    # test()
