import torch
import pandas as pd
import numpy as np
from data import data_augmentations
import os
import multiprocessing as mp
import random


class TenStreamDataset(torch.utils.data.Dataset):
    def __init__(self, cfg_data, cfg_transforms, cfg_augmentations, target_file, is_eval_set):
        super(TenStreamDataset).__init__()

        self.is_eval_set = is_eval_set
        self.allowed_views = cfg_data.allowed_views
        self.targets = pd.read_csv(os.path.abspath(target_file), sep=cfg_data.file_sep)
        if cfg_transforms.scale_output:
            self.targets['target'] = self.targets['target'].apply(lambda x: x / 100)
        self.filtered_targets = self.filter_incomplete_exams()
        if is_eval_set:
            self.generate_all_combinations()
        self.unique_exams = self.filtered_targets.drop_duplicates('us_id')['us_id']
        self.data_aug_img = data_augmentations.DataAugmentations(cfg_transforms, cfg_augmentations.img)
        self.data_aug_flow = data_augmentations.DataAugmentations(cfg_transforms, cfg_augmentations.flow)
        self.base_folder_img = cfg_data.data_folder_img
        self.base_folder_flow = cfg_data.data_folder_flow

    def __len__(self):
        return len(self.unique_exams)

    def __getitem__(self, index):
        exam = self.unique_exams.iloc[index]
        data_list = []
        for view in self.allowed_views:
            df = self.filtered_targets[self.filtered_targets['view'] == view].reset_index(drop=True)
            all_exam_indx = df[df['us_id'] == exam].index
            rndm_indx = random.choice(all_exam_indx)
            img, flow, target, uid = self.read_image_data(tuple(df.iloc[rndm_indx]))
            img = self.data_aug_img.transform_values(img).transpose(3, 0, 1, 2)
            data_list.append(img)
            flow = self.data_aug_flow.transform_values(flow).transpose(3, 0, 1, 2)
            data_list.append(flow)

        return data_list, np.expand_dims(target, axis=0).astype(np.float32), index, exam

    def load_data_into_mem(self):
        nprocs = mp.cpu_count()
        print(f"Number of CPU cores: {nprocs}")
        pool = mp.Pool(processes=nprocs)
        iterator = self.targets.itertuples(index=False, name=None)
        result = pool.map(self.read_image_data, iterator)
        pool.close()
        pool.join()
        data_list = []
        for r in result:
            data_list.append(r)
        self.data_frame = pd.DataFrame(data_list, columns=['img', 'flow', 'target', 'us_id'])
        print('All data loaded into memory')

    def read_image_data(self, data):
        uid, _, _, fps, hr, file_img, file_flow, target = data
        fp_img = os.path.join(os.path.join(self.base_folder_img, uid), file_img)
        fp_flow = os.path.join(os.path.join(self.base_folder_flow, uid), file_flow)
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

    def filter_incomplete_exams(self):
        unique_exams = self.targets.drop_duplicates('us_id')['us_id'].copy()
        view_arr = []
        for row in unique_exams:
            view_arr.append(all(elem in self.targets[self.targets['us_id'] == row]['view'].values for elem in self.allowed_views))
        filtered_ue = unique_exams[view_arr].copy()
        filtered_targets = self.targets[self.targets['us_id'].isin(filtered_ue)].copy().reset_index(drop=True)
        return filtered_targets



