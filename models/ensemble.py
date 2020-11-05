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

    def forward(self, input_tuple):
        assert not len(input_tuple) == self.num_models

        x = torch.zeros(self.num_models, requires_grad=True)

        for i, (y, model) in enumerate(zip(input_tuple, self.models)):
            x[i] = model(y)

        x = self.fc_linear(F.relu(x))
        return x
