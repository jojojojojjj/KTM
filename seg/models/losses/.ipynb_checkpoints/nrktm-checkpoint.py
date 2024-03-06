# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import torch
import torch.nn as nn
from .cwd import ChannelWiseDivergence
from .dist_loss import DISTLoss

def _expand_onehot_labels(labels,  target_shape, ignore_index,ignore):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_zeros(target_shape)
    if ignore_index==ignore:
        valid_mask = (labels >= 0) & (labels != ignore_index)
    else:
        valid_mask = (labels >= 0) & (labels != ignore_index)& (labels != ignore)
    inds = torch.nonzero(valid_mask, as_tuple=True)

    if inds[0].numel() > 0:
        if labels.dim() == 3:
            bin_labels[inds[0], labels[valid_mask], inds[1], inds[2]] = 1
        else:
            bin_labels[inds[0], labels[valid_mask]] = 1

    return bin_labels



class NRKTMLoss(nn.Module):

    def __init__(self,
                 loss_weight=1.0,
                 ignore_index=255,
                 ignore=255,
                 p_max=15,
                 p_min=-2,
                 tau=1.0,
                 kd='',
                 loss_name='loss_kd',
                 pooling=False,
                ):
        """Compute nrktm loss.

        Args:
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            ignore_index (int, optional): The label index to be ignored.
                Default: 255.
            ignore (int, optional): The label index to be ignored.
                Default: 255.
            p_max (float, optional):P_max of linear mapping.Defaults to 15.
            p_min (float, optional):P_min of linear mapping.Defaults to -2.
            tau (float, optional): temperature. Defaults to 1.0.
            loss_name (str, optional): Name of the loss item. If you want this
                loss item to be included into the backward graph, `loss_` must
                be the prefix of the name. Defaults to 'loss_kd'.
            kd (str):Name of the kd loss. You can choose cwd or dist.
            pooling (bool):Whether to downsample the prediction. Defaults to 'False'.
        """

        super().__init__()
        self.loss_weight= loss_weight
        self.p_max=p_max
        self.p_min=p_min
        self.tau=tau
        self.ignore= ignore
        self.ignore_index = ignore_index
        self._loss_name = loss_name
        self.kd=kd
        self.pooling=pooling

    def forward(self,
                pred,
                target,
                **kwargs
                ):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction, has a shape (b,c,h,w).
            target (torch.Tensor): The label of the prediction,
                shape (b,h,w).

        Returns:
            torch.Tensor: The calculated loss
        """
        num_classes = pred.shape[1]
        if self.pooling:
            pred = torch.nn.functional.interpolate(
                pred,
                (pred.shape[2]//2, pred.shape[3]//2),
                mode='nearest')
            target=target.unsqueeze(1).float()
            target = torch.nn.functional.interpolate(
                target,
                (pred.shape[2], pred.shape[3]),
                mode='nearest')
            target = target.squeeze(1).long()
        one_hot_target = target
        if (pred.shape != target.shape):
            pred_shape=pred.shape
            pred_gt = _expand_onehot_labels(target,pred_shape,self.ignore_index,self.ignore)
        assert pred.shape[1] != 0
        pred_gt=pred_gt*(self.p_max-self.p_min)+self.p_min
        if self.kd=='dist':
            KD=DISTLoss(tau=self.tau,loss_weight=self.loss_weight)
            loss=KD(pred, pred_gt)
        elif self.kd=='cwd':
            KD=ChannelWiseDivergence(tau=self.tau,loss_weight=self.loss_weight)
            loss=KD(pred, pred_gt)
        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
