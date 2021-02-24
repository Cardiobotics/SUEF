# Hinge loss for regression
import torch


class HingeLossRegression():
    def __init__(self, epsilon, reduction=None):
        self.epsilon = epsilon
        assert reduction in ['mean', 'sum', None]
        self.reduction = reduction

    def __call__(self, pred, target):
        if target.isnan().any() or pred.isnan().any():
            raise ValueError('Cannot calculate loss with NaN values. targets: {} preds: {}'.format(target, pred))
        loss = torch.clamp(torch.abs(target - pred)-self.epsilon, min=0)
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss