import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
mseloss = torch.nn.MSELoss()

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

class Crierition(nn.Module):
    def __init__(self, num_classes, matcher, class_weight, loss_weight):
        super().__init__()
        self.matcher = matcher
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.class_weight = class_weight

    def loss_reg(self, outputs, targets, indices, num_points):
        # with torch.no_grad():
        """ Regression loss """
        eps = 1e-8
        idx = self._get_src_permutation_idx(indices)
        outputs = outputs[0]
        src_points = outputs['pnt_coords'][idx]

        target_points = torch.cat([gt_points[J] for gt_points, (_, J) in zip(targets['gt_points'], indices)], dim=0)

        #print(target_points.size(),src_points.size())

        loss_pnt = F.mse_loss(src_points, target_points, reduction='none')
        loss_dict = {'loss_reg': loss_pnt.sum() / (num_points + eps)}
        #print(loss_dict,num_points)
        return loss_dict

    def loss_cls(self, outputs, targets, indices, num_points):
        # with torch.no_grad():
        """Classification loss """
        idx = self._get_src_permutation_idx(indices)
        outputs = outputs[0]
        src_logits = outputs['cls_logits']

        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.long, device=src_logits.device)
        target_classes_o = torch.cat([cls[J] for cls, (_, J) in zip(targets['gt_labels'], indices)])
        target_classes[idx] = target_classes_o

        loss_cls = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.class_weight)
        loss_dict = {'loss_cls': loss_cls}
        return loss_dict

    def get_loss(self, loss, outputs, targets, indices, num_points, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'points': self.loss_points,
            'consist': self.loss_consist,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_points, **kwargs)

    def loss_labels(self, outputs, targets, indices, num_points):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'cls_logits' in outputs
        src_logits = outputs['cls_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(targets["gt_labels"], indices)])
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.class_weight)
        #loss_gce = self.gce_loss(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        #losses = {'loss_ce': loss_gce}

        return losses

    def loss_points(self, outputs, targets, indices, num_points):
        assert 'pnt_coords' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pnt_coords'][idx]
        target_points = torch.cat([t[i] for t, (_, i) in zip(targets['gt_points'], indices)], dim=0)

        loss_bbox = F.mse_loss(src_points, target_points, reduction='none')

        losses = {}
        losses['loss_point'] = loss_bbox.sum() / num_points

        return losses

    def loss_consist(self, outputs, targets, indices, num_points):

        output1 = {'cls_logits': outputs[0]['cls_logits'], 'pnt_coords': outputs[0]['pnt_coords']}
        indices1 = self.matcher(output1, targets)
        num_points = sum(targets['gt_nums'])
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(output1.values())).device)

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_points)
        num_boxes = torch.clamp(num_points / get_world_size(), min=1).item()
        loss_point1 = self.get_loss("points", output1, targets, indices1, num_boxes)
        loss_label1 = self.get_loss("labels", output1, targets, indices1, num_boxes)

        output1_style = {'cls_logits': outputs[1]['cls_logits'], 'pnt_coords': outputs[1]['pnt_coords']}
        indices1 = self.matcher(output1_style, targets)
        num_points = sum(targets['gt_nums'])
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(output1_style.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_points)
        num_boxes = torch.clamp(num_points / get_world_size(), min=1).item()
        loss_point1_style = self.get_loss("points", output1_style, targets, indices1, num_boxes)
        loss_label1_style = self.get_loss("labels", output1_style, targets, indices1, num_boxes)

        loss_task2 = (loss_point1['loss_point'] + loss_point1_style ['loss_point']) / 2.0
        loss_task1 = (loss_label1['loss_ce'] + loss_label1_style['loss_ce']) / 2.0

        loss_concent_consist1 = 0.1 * mseloss(output1['cls_logits'], output1_style['cls_logits'])
        loss_concent_consist2 = 0.1 * mseloss(output1['pnt_coords'], output1_style['pnt_coords'])


        loss_content = (loss_task1 + loss_concent_consist1) + 0.0002 * (loss_task2 + loss_concent_consist2)
        losses = {'loss_consist': loss_content}
        return losses

    @staticmethod
    def _get_src_permutation_idx(indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs = [outputs, outputs]

        indices = self.matcher(outputs[0], targets)

        num_points = sum(targets['gt_nums'])
        num_points = torch.as_tensor(num_points, dtype=torch.float)

        losses = {}
        loss_map = {
            'loss_reg': self.loss_reg,
            'loss_cls': self.loss_cls,
            # 'loss_consist': self.loss_consist,
        }

        for loss_func in loss_map.values():
            losses.update(loss_func(outputs, targets, indices, num_points))
        weight_dict = self.loss_weight

        losses = torch.stack([losses[k] * weight_dict[k] for k in weight_dict if k in losses])

        # losses = [losses[k] * weight_dict[k] for k in weight_dict if k in losses]
        # add_loss = - (outputs['add_pred'].softmax(-1).log() * targets['cell_ratios']).sum(1).mean()
        # losses.append(add_loss)
        # losses = torch.stack(losses)
        return losses


def build_criterion(rank, matcher, args):
    #class_weight = torch.Tensor([1,1,1,4,1],dtype=torch.float,device=f'cuda:{rank}')
    class_weight = torch.ones(args.num_classes + 1, dtype=torch.float, device=f'cuda:{rank}')
    class_weight[-2] = 5
    class_weight[-1] = args.eos_coef

    loss_weight = {'loss_reg': args.reg_loss_coef, 'loss_cls': args.cls_loss_coef, "loss_consist": 1}
    # loss_weight = {'loss_reg': 0, 'loss_cls': 0, "loss_consist": 1}
    # loss_weight = {"loss_consist": 1}
    return Crierition(
        args.num_classes,
        matcher,
        class_weight=class_weight,
        loss_weight=loss_weight
    )
