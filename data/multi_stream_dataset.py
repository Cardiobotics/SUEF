import torch
import pandas as pd
import numpy as np
from data import data_augmentations
import os
import multiprocessing as mp
import random
import itertools


class MultiStreamDataset(torch.utils.data.Dataset):
    def __init__(self, cfg_data, cfg_transforms, cfg_augmentations, target_file, is_eval_set):
        super(MultiStreamDataset).__init__()

        self.is_eval_set = is_eval_set
        self.allowed_views = cfg_data.allowed_views
        self.targets = pd.read_csv(os.path.abspath(target_file), sep=cfg_data.file_sep)
        if cfg_transforms.scale_output:
            self.targets['target'] = self.targets['target'].apply(lambda x: x / 100)
        self.unique_exams = self.targets.drop_duplicates('us_id')[['us_id', 'target']]
        self.data_aug_img = data_augmentations.DataAugmentations(cfg_transforms, cfg_augmentations.img)
        self.data_aug_flow = data_augmentations.DataAugmentations(cfg_transforms, cfg_augmentations.flow)
        self.base_folder_img = cfg_data.data_folder_img
        self.base_folder_flow = cfg_data.data_folder_flow
        if is_eval_set:
            self.generate_all_combinations()

    def __len__(self):
        if self.is_eval_set:
            return len(self.targets)
        else:
            return len(self.unique_exams)

    def __getitem__(self, index):
        if self.is_eval_set:
            df = self.targets.iloc[index]
            exam = df['us_id']
            target = df['target']
            data_list = []
            for view in self.allowed_views:
                iid = df['instance_id_' + str(view)]
                if iid is None or exam is None or target is None:
                    img = np.zeros((1, self.data_aug_img.transforms.target_length,
                                    self.data_aug_img.transforms.target_height,
                                    self.data_aug_img.transforms.target_width), dtype=np.float32)
                    data_list.append(img)
                    flow = np.zeros((2, self.data_aug_flow.transforms.target_length,
                                     self.data_aug_flow.transforms.target_height,
                                     self.data_aug_flow.transforms.target_width), dtype=np.float32)
                    data_list.append(flow)
                else:
                    fps = df['fps_' + str(view)]
                    hr = df['hr_' + str(view)]
                    file_img = df['filename_img_' + str(view)]
                    file_flow = df['filename_flow_' + str(view)]
                    img, flow, _, _ = self.read_image_data(tuple((exam, 0, 0, fps, hr, file_img, file_flow, target)))
                    img = self.data_aug_img.transform_values(img).transpose(3, 0, 1, 2)
                    data_list.append(img)
                    flow = self.data_aug_flow.transform_values(flow).transpose(3, 0, 1, 2)
                    data_list.append(flow)
        else:
            exam = self.unique_exams.iloc[index].us_id
            target = self.unique_exams.iloc[index].target
            data_list = []
            for view in self.allowed_views:
                df = self.targets[self.targets['view'] == view].reset_index(drop=True)
                all_exam_indx = df[df['us_id'] == exam].index
                if len(all_exam_indx) > 0:
                    rndm_indx = random.choice(all_exam_indx)
                    img, flow, _, _ = self.read_image_data(tuple(df.iloc[rndm_indx]))
                    img = self.data_aug_img.transform_values(img).transpose(3, 0, 1, 2)
                    data_list.append(img)
                    flow = self.data_aug_flow.transform_values(flow).transpose(3, 0, 1, 2)
                    data_list.append(flow)
                else:
                    img = np.zeros((1, self.data_aug_img.transforms.target_length,
                                       self.data_aug_img.transforms.target_height,
                                       self.data_aug_img.transforms.target_width), dtype=np.float32)
                    data_list.append(img)
                    flow = np.zeros((2, self.data_aug_flow.transforms.target_length,
                                        self.data_aug_flow.transforms.target_height,
                                        self.data_aug_flow.transforms.target_width), dtype=np.float32)
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

    def combinate(self, items, size=4):
        for cats in itertools.combinations(items.keys(), size):
            cat_items = [[products for products in items[cat]] for cat in cats]
            for x in itertools.product(*cat_items):
                yield zip(cats, x)

    def generate_all_combinations(self):
        all_generated_combinations = []
        for ue in self.unique_exams.itertuples():
            new_dict = {}
            t = self.targets[self.targets['us_id'] == ue.us_id]
            for view in self.allowed_views:
                ue_data = t[t.view == view][['instance_id', 'fps', 'hr', 'filename_img', 'filename_flow']].values.tolist()
                if len(ue_data) > 0:
                    new_dict[view] = ue_data
                else:
                    new_dict[view] = [[None, None, None, None, None]]
            ue_combs = self.combinate(new_dict)

            for uec in ue_combs:
                pd_dict = {'us_id': ue.us_id, 'target': ue.target}
                for view, data in uec:
                    pd_dict['instance_id_' + str(view)] = data[0]
                    pd_dict['fps_' + str(view)] = data[1]
                    pd_dict['hr_' + str(view)] = data[2]
                    pd_dict['filename_img_' + str(view)] = data[3]
                    pd_dict['filename_flow_' + str(view)] = data[4]
                all_generated_combinations.append(pd_dict)
        self.targets = pd.DataFrame(all_generated_combinations)
