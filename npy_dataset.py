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


class NPYDataset(torch.utils.data.Dataset):
    def __init__(self, folder, target_file, transforms, augmentations, file_sep):
        super(NPYDataset).__init__()

        self.targets = pd.read_csv(os.path.abspath(target_file), sep=file_sep)
        if transforms.scale_output:
            self.targets['target'] = self.targets['target'].apply(lambda x: x / 100)
        self.data_aug = data_transforms.DataAugmentations(transforms, augmentations)
        self.data_list = []
        self.base_folder = folder
        self.load_data_into_mem()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img, target = self.data_list[index]
        img = self.data_aug.transform_values(img)
        return img.transpose(3, 0, 1, 2), target

    def load_data_into_mem(self):
        nprocs = mp.cpu_count()
        print(f"Number of CPU cores: {nprocs}")
        pool = mp.Pool(processes=nprocs)
        iterator = self.targets.itertuples(index=False, name=None)
        result = pool.map(self.read_image_data, iterator)
        pool.close()
        pool.join()
        for r in result:
            self.data_list.append(r)
        print('All data loaded into memory')

    def read_image_data(self, data):
        fps = data[3]
        target = data[4]
        file = data[5]
        fp = os.path.join(self.base_folder, file)
        try:
            img = np.load(fp, allow_pickle=True)
            img = self.data_aug.transform_size(img, fps)
            if target is None:
                raise ValueError("Target is None")
            if img is None:
                raise ValueError("Img is None")
            return img, target
        except Exception as e:
            print("Failed to get item for File: {} with exception: {}".format(file, e))
