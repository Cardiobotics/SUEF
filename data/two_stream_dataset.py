import torch
import pandas as pd
import numpy as np
import data_transforms
import os
import multiprocessing as mp
import random


class TwoStreamDataset(torch.utils.data.Dataset):
    def __init__(self, cfg_data, cfg_transforms, cfg_augmentations, target_file):
        super(TwoStreamDataset).__init__()

        self.targets = pd.read_csv(os.path.abspath(target_file), sep=cfg_data.file_sep)
        if cfg_transforms.scale_output:
            self.targets['target'] = self.targets['target'].apply(lambda x: x / 100)
        self.targets = self.targets[self.targets['view'].isin(cfg_data.allowed_views)].reset_index(drop=True)
        self.unique_exams = self.targets.drop_duplicates('us_id')['us_id'].copy()
        self.data_aug_img = data_transforms.DataAugmentations(cfg_transforms.img, cfg_augmentations.img)
        self.data_aug_flow = data_transforms.DataAugmentations(cfg_transforms.flow, cfg_augmentations.flow)
        self.base_folder = cfg_data.data_folder
        self.data_in_mem = cfg_data.data_in_mem
        if self.data_in_mem:
            self.data_list = []
            self.load_data_into_mem()

    def __len__(self):
        return len(self.unique_exams)

    def __getitem__(self, index):
        exam = self.unique_exams.iloc[index]
        if self.data_in_mem:
            all_exam_indx = self.data_frame[self.data_frame['us_id'] == exam].index
            rndm_indx = random.choice(all_exam_indx)
            img = self.data_frame.iloc[rndm_indx].img
            flow = self.data_frame.iloc[rndm_indx].flow
            target = self.data_frame.iloc[rndm_indx].target
            uid = self.data_frame.iloc[rndm_indx].us_id
        else:
            all_exam_indx = self.targets[self.targets['us_id'] == exam].index
            rndm_indx = random.choice(all_exam_indx)
            img, flow, target, uid = self.read_image_data(tuple(self.targets.iloc[rndm_indx]))
        img = self.data_aug_img.transform_values(img)
        flow = self.data_aug_flow.transform_values(flow)
        return (img.transpose(3, 0, 1, 2), flow.transpose(3, 0, 1, 2)), np.expand_dims(target, axis=0).astype(np.float32), index, uid

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
        fp_img = os.path.join(os.path.join(self.base_folder, uid), file_img)
        fp_flow = os.path.join(os.path.join(self.base_folder, uid), file_flow)
        try:
            if target is None:
                raise ValueError("Target is None")
            # Process img
            img = np.load(fp_img, allow_pickle=True)
            img = self.data_aug_img.transform_size(img, fps, hr)
            if img is None:
                raise ValueError("Img is None")

            # Process flow
            flow = np.load(fp_flow, allow_pickle=True)
            flow = self.data_aug_flow.transform_size(flow, fps, hr)
            if flow is None:
                raise ValueError("Flow is None")
            return img, flow, target, uid
        except Exception as e:
            print("Failed to get item for img file: {} and flow file: {} with exception: {}".format(file_img, file_flow, e))
