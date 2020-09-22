import torch
import pandas as pd
import numpy as np
import pydicom
import data_transforms
import os
import random
from PIL import Image
import multiprocessing as mp
from pympler import asizeof



class DCMDataset(torch.utils.data.Dataset):
    def __init__(self, view_file, target_file, transforms, augmentations, file_sep):
        super(DCMDataset).__init__()

        self.targets = pd.read_csv(os.path.abspath(target_file), sep=file_sep)
        if transforms.scale_output:
            self.targets['target'] = self.targets['target'].apply(lambda x: x / 100)
        self.view_files = pd.read_csv(os.path.abspath(view_file), sep=file_sep)
        self.data_aug = data_transforms.DataAugmentations(transforms, augmentations)
        self.pd_data = pd.merge(self.view_files, self.targets, on='us_id')
        self.data_list = []
        self.load_data_into_mem()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img, target = self.data_list[index]
        img = self.data_aug.transform_values(img)
        return img, target

    def old_getitem(self, index):
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
            img = self.data_aug.transform(dcm_data)

        except ValueError as e:
            print("Target UID: {}, Files: {}, Failed with exception: {} ".format(uid, uid_files, e))
        except Exception as e:
            print("Failed to get item for UID: {}, Files: {}, Row: {} , with exception: {}".format(uid, uid_files, row, e))

        return img, target

    def load_data_into_mem(self):
        nprocs = mp.cpu_count()
        print(f"Number of CPU cores: {nprocs}")
        pool = mp.Pool(processes=nprocs)
        iterator = self.pd_data.itertuples(index=False, name=None)
        result = pool.map(self.read_image_data, iterator)
        print('Conversion to grayscale complete')
        pool.close()
        pool.join()
        for r in result:
            self.data_list.append(r)

    def read_image_data(self, data):
        file = data[3]
        target = data[4]
        try:
            dcm_data = pydicom.read_file(file)
            fps = dcm_data.RecommendedDisplayFrameRate
            img = dcm_data.pixel_array
            img = self.data_aug.transform_size(img, fps)
            if target is None:
                raise ValueError("Target is None")
            if img is None:
                raise ValueError("Img is None")
            #print('Size of img is: {}'.format(asizeof.asizeof(img)))
            return img, target
        except Exception as e:
            print("Failed to get item for File: {} with exception: {}".format(file, e))
