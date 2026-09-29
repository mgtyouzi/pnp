"""
Mostly copy-paste from DETR (https://github.com/facebookresearch/detr).
"""
import numpy as np
import torch
from torch import nn
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_point, cost_class):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the foreground object
            cost_point: This is the relative weight of the L1 error of the points coordinates in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_point = cost_point
        self.top_k = 1

    @torch.no_grad()
    def forward(self, outputs, targets):
        # bs = len(targets['gt_nums'])
        bs, num_queries = outputs["pnt_coords"].shape[:2]

        # Compute the regression cost.
        out_coords = outputs["pnt_coords"].flatten(0, 1)
        cost_point = torch.cdist(out_coords, torch.cat(targets['gt_points']), p=2, compute_mode='donot_use_mm_for_euclid_dist')
        # cost_point = torch.cdist(out_coords, torch.cat(targets['gt_points']), p=2)

        # Compute the classification cost.
        out_prob = outputs["cls_logits"].flatten(0, 1).softmax(-1)
        cost_class = - (out_prob[:, torch.cat(targets['gt_labels'])] + 1e-8).log()

        # Final cost matrix.
        C = self.cost_point * cost_point + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).detach().cpu()

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(targets['gt_nums'], -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

        # indices = [self.assign(outputs['pnt_coords'][i],
        #                        outputs['cls_logits'][i].softmax(-1),
        #                        targets['gt_points'][i],
        #                        targets['gt_labels'][i]) for i in range(bs)]
        # return indices

    @torch.no_grad()
    def assign(self,
               pred_point,
               pred_prob,
               gt_points,
               gt_labels,
               eps=1e-8):
        cost_point = torch.cdist(pred_point, gt_points, p=2, compute_mode='donot_use_mm_for_euclid_dist')
        cost_class = - pred_prob[:, gt_labels]
        # cost_class = - (pred_prob[:, gt_labels] + eps).log()

        cost = self.cost_point * cost_point + self.cost_class * cost_class
        cost = cost.cpu().numpy()

        n, m = len(pred_point), len(gt_points)
        index = torch.arange(n)
        valid_flag = torch.ones((n,), dtype=torch.bool)
        assign_matrix = torch.zeros((n, m), dtype=torch.bool)

        for _ in range(self.top_k):
            new_cost = cost[valid_flag]
            new_index = index[valid_flag]

            matched_row_inds, matched_col_inds = linear_sum_assignment(new_cost)
            matched_row_inds = torch.from_numpy(matched_row_inds)
            matched_col_inds = torch.from_numpy(matched_col_inds)

            matched_row_inds = new_index[matched_row_inds]  # convert to global index

            assign_matrix[matched_row_inds, matched_col_inds] = True
            valid_flag[matched_row_inds] = False

        return torch.where(assign_matrix)


def build_matcher(args):
    matcher = HungarianMatcher(
        cost_point=args.set_cost_point,
        cost_class=args.set_cost_class
    )
    return matcher
