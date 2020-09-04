import torch
import pandas as pd
import pydicom
import cnn_data_aug
import os
import random


class DCMDataset(torch.utils.data.Dataset):
    def __init__(self, view_file, target_file, allowed_views, file_sep=';', transform_flags={}):
        super(DCMDataset).__init__()

        self.targets = pd.read_csv(os.path.abspath(target_file), sep=file_sep)
        view_files = pd.read_csv(os.path.abspath(view_file), sep=file_sep)
        self.files = view_files[view_files['prediction'].isin(allowed_views)].copy()
        self.data_aug = cnn_data_aug.DataAugmentations(transform_flags)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        uid = self.targets.iloc[index]['us_id']
        target = self.targets.iloc[index]['target']
        try:
            uid_files = self.files[self.files['us_id'] == uid]
            if len(uid_files) < 1:
                raise ValueError("No matching views for that examination")
            row = uid_files.iloc[random.randint(0, len(uid_files) - 1)]
            dcm_data = pydicom.read_file(os.path.abspath(row['file']))
            img = dcm_data.pixel_array
            if not dcm_data.InstanceNumber == row['instance_id'] and dcm_data.PatientID == uid:
                raise ValueError("InstanceID or PatientID not matching")
            if self.data_aug:
                img = self.data_aug.transform(img)
        except ValueError as e:
            print("Target UID: {}, Files: {}, Failed with exception: {} ".format(uid, uid_files, e))
        except Exception as e:
            print("Failed to get item for UID: {}, Files: {} , with exception: {}".format(uid, uid_files, e))

        return img, target
