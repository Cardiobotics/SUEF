from typing import Any

import torch.nn as nn
import torch
import torch.nn.functional as F


class FlexEnsemble(nn.Module):
    '''
    Ensemble for combining multiple models.
    If the models are already trained, all layers except the last fc_linear must be frozen.
    '''

    def __init__(self, *args):
        super(FlexEnsemble, self).__init__()
        self.models = []

        for model in args:
            self.models.append(model)
        self.num_models = len(args)

        self.fc_linear = nn.Linear(self.num_models, 1)

    def forward(self, *args):
        assert not len(args) == self.num_models

        output = []

        for y, model in zip(args, self.models):
            output.append(model(y))
        x = torch.cat(output, dim=1)

        x = self.fc_linear(F.relu(x))
        return x
