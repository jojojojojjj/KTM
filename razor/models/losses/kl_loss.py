import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.utils import resize

class CriterionKD(nn.Module):
    '''
    knowledge distillation loss
    '''

    def __init__(self,
                 tau: float = 1.0,
                 sigmoid: bool = True,
                 loss_weight: float = 1.0):
        super(CriterionKD, self).__init__()
        self.temperature = tau
        self.sigmoid = sigmoid
        self.loss_weight = loss_weight
        if self.sigmoid:
            self.criterion_kd = torch.nn.MSELoss()
        else:
            self.criterion_kd = torch.nn.KLDivLoss()

    def forward(self, preds_S, preds_T):
        preds_S = resize(
            input=preds_S,
            size=preds_T.shape[2:],
            mode='bilinear',
            align_corners=False)
        preds_T.detach()
        if self.sigmoid:
            loss = self.criterion_kd(
                torch.sigmoid(preds_S),
                torch.sigmoid(preds_T))
        else:
            loss = self.criterion_kd(
                F.log_softmax(preds_S / self.temperature, dim=1),
                F.softmax(preds_T / self.temperature, dim=1))
        return self.loss_weight * loss