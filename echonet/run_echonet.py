import math
import os
import time

import numpy as np
import sklearn.metrics
import torch
import torchvision
import tqdm

def run_epoch(model, dataloader, train, optim, device):
    """Run one epoch of training/evaluation for segmentation.
    Args:
        model (torch.nn.Module): Model to train/evaulate.
        dataloder (torch.utils.data.DataLoader): Dataloader for dataset.
        train (bool): Whether or not to train model.
        optim (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to run on
    """

    model.train(train)

    res = []
    yhat = []
    y = []

    with torch.set_grad_enabled(train):
        for (x, targets) in tqdm.tqdm(dataloader):

            y.append(targets.numpy())
            x = x.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True, dtype=torch.float)
            preds = model(x)
            loss = torch.nn.functional.mse_loss(preds.view(-1), targets)
            yhat.append(preds.view(-1).to("cpu").detach().numpy())

            if train:
                optim.zero_grad()
                loss.backward()
                optim.step()
            res.append(loss.item())

    yhat = np.concatenate(yhat)
    y = np.concatenate(y)
    r2_score = sklearn.metrics.r2_score(y, yhat)
    return np.mean(res), r2_score
