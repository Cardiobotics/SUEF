import torch.nn as nn
import torch
import torch.nn.functional as F


class CNNEnsemble(nn.Module):
    '''
    Ensemble for combining multiple models.
    If the models are already trained, all layers except the last fc_linear must be frozen.
    '''
    def __init__(self, model1, model2, model3, model4, model5):
        super(CNNEnsemble, self).__init__()

        self.view_model_1 = model1
        self.view_model_2 = model2
        self.view_model_3 = model3
        self.view_model_4 = model4
        self.view_model_5 = model5

        self.fc_linear = nn.Linear(5, 1)

    def forward(self, v1, v2, v3, v4, v5):
        x1 = self.view_model_1(v1)
        x2 = self.view_model_1(v2)
        x3 = self.view_model_1(v3)
        x4 = self.view_model_1(v4)
        x5 = self.view_model_1(v5)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.fc_linear(F.relu(x))
        return x
