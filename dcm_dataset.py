import torch
import pandas as pd
import numpy as np
import pydicom
import data_transforms
import os
import random
import config


class DCMDataset(torch.utils.data.Dataset):
    def __init__(self, view_file, target_file, t_settings, file_sep=';'):
        super(DCMDataset).__init__()

        self.targets = pd.read_csv(os.path.abspath(target_file), sep=file_sep)
        if t_settings['scale_output']:
            self.targets['target'] = self.targets['target'].apply(lambda x: x / 100)
        view_files = pd.read_csv(os.path.abspath(view_file), sep=file_sep)
        self.files = view_files[view_files['prediction'].isin(config.allowed_views)].copy()
        self.data_aug = data_transforms.DataAugmentations(t_settings)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        uid = self.targets.iloc[index]['us_id']
        target = self.targets.iloc[index]['target'].astype(np.float32)
        try:
            uid_files = self.files[self.files['us_id'] == uid]
            if len(uid_files) < 1:
                raise ValueError("No matching views for that examination")
            row = uid_files.iloc[random.randint(0, len(uid_files) - 1)]
            file = row['file']
            dcm_data = pydicom.read_file(file)
            img = dcm_data.pixel_array
            if not dcm_data.InstanceNumber == row['instance_id'] and dcm_data.PatientID == uid:
                raise ValueError("InstanceID or PatientID not matching")
            if self.data_aug:
                img = self.data_aug.transform(dcm_data)
        except ValueError as e:
            print("Target UID: {}, Files: {}, Failed with exception: {} ".format(uid, uid_files, e))
        except Exception as e:
            print("Failed to get item for UID: {}, Files: {} , with exception: {}".format(uid, uid_files, e))

        return img, target
