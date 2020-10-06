import torch
import pandas as pd
import numpy as np
import data_transforms
import os
import multiprocessing as mp


class NPYDataset(torch.utils.data.Dataset):
    def __init__(self, cfg_data, cfg_transforms, cfg_augmentations, target_file):
        super(NPYDataset).__init__()

        self.targets = pd.read_csv(os.path.abspath(target_file), sep=cfg_data.file_sep)
        if cfg_transforms.scale_output:
            self.targets['target'] = self.targets['target'].apply(lambda x: x / 100)
        self.targets = self.targets[self.targets['view'].isin(cfg_data.allowed_views)]
        self.data_aug = data_transforms.DataAugmentations(cfg_transforms, cfg_augmentations)
        self.data_type = cfg_data.type
        self.data_list = []
        self.base_folder = cfg_data.data_folder
        self.load_data_into_mem()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img, target = self.data_list[index]
        img = self.data_aug.transform_values(img)
        return img.transpose(3, 0, 1, 2), np.expand_dims(target, axis=0).astype(np.float32), index

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
        uid, _, _, fps, hr, file_img, file_flow, target = data
        if self.data_type == 'flow':
            file = file_flow
        elif self.data_type == 'img':
            file = file_img
        fp = os.path.join(os.path.join(self.base_folder, uid), file)
        try:
            img = np.load(fp, allow_pickle=True)
            img = self.data_aug.transform_size(img, fps, hr)
            if target is None:
                raise ValueError("Target is None")
            if img is None:
                raise ValueError("Img is None")
            return img, target
        except Exception as e:
            print("Failed to get item for File: {} with exception: {}".format(file, e))
