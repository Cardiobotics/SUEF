import torch
import numpy as np

class OrdinalRegressionAT:
    def __init__(self, sample_weights=None, reduction=None):
        assert reduction in ['mean', 'sum', None]
        self.reduction = reduction
        self.sample_weights = sample_weights

    def __call__(self, output, y, thresholds):
        output = output.squeeze()
        y = y.squeeze().long()
        # All unique classes
        classes = torch.unique(y)
        # Number of classes
        n_class = len(thresholds) + 1
        # Loss forward difference
        loss_fd = torch.ones((n_class, n_class - 1)).cuda()[y]
        Alpha = thresholds[:, None] - output
        S = torch.sign(torch.arange(n_class - 1).cuda()[:, None] - y + 0.5)
        err = loss_fd.T * self.log_loss(S * Alpha)
        if self.sample_weights is not None:
            weights = self.sample_weights[y]
            err *= weights
        loss = torch.sum(err, dim=0)
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

    def log_loss(self, Z):
        # stable computation of the logistic loss
        idx = Z > 0
        out = torch.zeros_like(Z)
        out[idx] = torch.log(1 + torch.exp(-Z[idx]))
        out[~idx] = (-Z[~idx] + torch.log(1 + torch.exp(Z[~idx])))
        return out

def get_ORAT_labels(outputs, thresholds):
    tc = thresholds.detach().cpu().numpy()
    tcs = np.sort(tc)
    if not (tcs == tc).all():
        raise ValueError("Thresholds not sorted from lowest to highest, cannot compute labels")
    labels = np.zeros(outputs.shape)
    for i, t in enumerate(tc):
        for j, o in enumerate(outputs):
            # Edge case start
            if i == 0:
                if o < t:
                    labels[j] = i
            # Edge case end
            if i == (len(tc) - 1):
                if o < t and o >= tc[i-1]:
                    labels[j] = i
                elif o >= t:
                    labels[j] = i+1
            # All other cases
            else:
                if o < t and o >= tc[i-1]:
                    labels[j] = i
    return labels
