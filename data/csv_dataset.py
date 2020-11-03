import torch
import pandas as pd
import numpy as np


class CSVDataset(torch.utils.data.Dataset):
    def __init__(self, flow_3c, img_3c, flow_4c, img_4c):
        super(CSVDataset).__init__()

        pd_flow_3c = pd.read_csv(flow_3c, sep=';')
        pd_img_3c = pd.read_csv(img_3c, sep=';')
        pd_flow_4c = pd.read_csv(flow_4c, sep=';')
        pd_img_4c = pd.read_csv(img_4c, sep=';')

        data = pd_flow_3c.merge(pd_img_3c, on=['us_id', 'target'], how='inner')
        data = data.rename(columns={"pred_x": "pred_2_flow", 'pred_y': 'pred_2_img'})
        data = data.merge(pd_flow_4c, on=['us_id', 'target'], how='inner')
        data = data.rename(columns={"pred": "pred_4_flow"})
        data = data.merge(pd_img_4c, on=['us_id', 'target'], how='inner')
        data = data.rename(columns={"pred": "pred_4_img"})
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        target = row['target']
        input = np.array([row['pred_2_flow'], row['pred_2_img'], row['pred_4_flow'], row['pred_4_img']]).astype(np.float32)
        return input,  np.expand_dims(target, axis=0).astype(np.float32)