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
        view_files = pd.read_csv(os.path.abspath(target_file), sep=file_sep)
        self.files = view_files[view_files['prediction'] in allowed_views].copy()
        self.transform = cnn_data_aug.DataAugmentations(transform_flags)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        uid = self.targets.iloc[index]['us_id']
        target = self.targets.iloc[index]['target']
        uid_files = self.files[self.files['us_id'] == uid]
        row = uid_files.iloc[random.randint(0, len(uid_files))]
        try:
            dcm_data = pydicom.read_file(os.path.abspath(row['file']))
            img = dcm_data.pixel_array
            if not dcm_data.InstanceNumber == row['instance_id'] and dcm_data.PatientID == uid:
                raise ValueError("InstanceID or PatientID not matching")
        except ValueError as e:
            print("File: {}, Target UID: {}, DCM UID: {}, Target InstID: {}, DCM InstID: {}, Failed with "
                  "exception: {} ".format(row['file'], uid, dcm_data.PatientID, row['instance_id'],
                                          dcm_data.InstanceNumber, e))
        except Exception as e:
            print("File: {}, failed to read with exception: {}".format(row['file'], e))

        if self.transform:
            img = self.transform(img)

        return img, target