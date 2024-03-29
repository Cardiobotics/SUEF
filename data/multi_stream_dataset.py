import torch
import pandas as pd
import numpy as np
from data import data_augmentations
import os
import multiprocessing as mp
import random
import itertools
import time


class MultiStreamDataset(torch.utils.data.Dataset):
    def __init__(self, cfg_data, cfg_transforms, cfg_augmentations, target_file, is_eval_set, minimum_validation):
        super(MultiStreamDataset).__init__()
        assert not (cfg_data.data_in_mem and cfg_data.preprocessed_data_on_disk)

        self.minimum_validation = minimum_validation
        self.is_eval_set = is_eval_set
        self.allowed_views = cfg_data.allowed_views
        self.data_in_mem = cfg_data.data_in_mem
        self.preprocessed_data_on_disk = cfg_data.preprocessed_data_on_disk
        self.targets = pd.read_csv(os.path.abspath(target_file), sep=cfg_data.file_sep)
        self.rwave_only = cfg_transforms.rwave_data_only
        if self.rwave_only:
            self.targets = self.targets[self.targets['rwaves'].notnull()]
        self.targets = self.targets.replace({np.nan: None})
        if cfg_transforms.scale_output:
            self.targets['target'] = self.targets['target'].apply(lambda x: x / 100)
        self.unique_exams = self.targets.drop_duplicates('us_id')[['us_id', 'target']]
        self.data_aug_img = data_augmentations.DataAugmentations(cfg_transforms, cfg_augmentations.img)
        self.data_aug_flow = data_augmentations.DataAugmentations(cfg_transforms, cfg_augmentations.flow)
        self.base_folder_img = cfg_data.data_folder_img
        self.base_folder_flow = cfg_data.data_folder_flow
        self.only_use_complete = cfg_data.only_use_complete_exams
        if self.only_use_complete:
            self.filter_incomplete_exams()
        if self.is_eval_set:
            self.targets_combinations = self.generate_all_combinations()
        if self.data_in_mem:
            self.load_data_into_mem()
        if self.preprocessed_data_on_disk:
            self.temp_folder_img = cfg_data.temp_folder_img
            self.temp_folder_flow = cfg_data.temp_folder_flow
            # Turn of the flag to enable transforms while saving to disk
            self.preprocessed_data_on_disk = False
            self.load_data_to_disk()
            self.base_folder_img = self.temp_folder_img
            self.base_folder_flow = self.temp_folder_flow
            # Enable it again so that transforms are disabled when reading from dataset.
            self.preprocessed_data_on_disk = True
        #Data consistensy checks
        if self.is_eval_set:
            if self.targets_combinations['target'].isnull().any():
                null_df = self.targets_combinations[self.targets_combinations['target'].isnull()]
                raise ValueError('Null values in targets: {}'.format(null_df))

    def __len__(self):
        if self.is_eval_set:
            return len(self.targets_combinations)
        else:
            return len(self.unique_exams)

    def __getitem__(self, index):
        data_list = []
        iid_list = []
        if self.is_eval_set:
            if self.data_in_mem:
                df = self.targets_combinations.iloc[index]
                exam = df['us_id']
                target = df['target']
                for view in self.allowed_views:
                    iid = df['instance_id_' + str(view)]
                    if np.isnan(iid):
                        img, flow = self.generate_dummy_data()
                        data_list.append(img)
                        data_list.append(flow)
                        iid_list.append(-1)
                    else:
                        row = self.data_frame[(self.data_frame['us_id'] == exam) & (self.data_frame['view'] == view) & (self.data_frame['iid'] == iid)]
                        img = row['img']
                        img = self.data_aug_img.transform_values(img).transpose(3, 0, 1, 2)
                        data_list.append(img)
                        flow = row['flow']
                        flow = self.data_aug_flow.transform_values(flow).transpose(3, 0, 1, 2)
                        data_list.append(flow)
                        iid_list.append(iid)
            else:
                df = self.targets_combinations.iloc[index]
                exam = df['us_id']
                target = df['target']
                for view in self.allowed_views:
                    iid = df['instance_id_' + str(view)]
                    if np.isnan(iid):
                        img, flow = self.generate_dummy_data()
                        data_list.append(img)
                        data_list.append(flow)
                        iid_list.append(-1)
                    else:
                        fps = df['fps_' + str(view)]
                        hr = df['hr_' + str(view)]
                        file_img = df['filename_img_' + str(view)]
                        file_flow = df['filename_flow_' + str(view)]
                        if self.rwave_only:
                            rwaves = df['rwaves_' + str(view)]
                        else:
                            rwaves = None
                        img, flow, _, _, _, _ = self.read_image_data(tuple((exam, 0, 0, fps, hr, file_img, file_flow, target, rwaves)))
                        img = self.data_aug_img.transform_values(img).transpose(3, 0, 1, 2)
                        data_list.append(img)
                        flow = self.data_aug_flow.transform_values(flow).transpose(3, 0, 1, 2)
                        data_list.append(flow)
                        iid_list.append(iid)
        else:
            exam = self.unique_exams.iloc[index].us_id
            target = self.unique_exams.iloc[index].target
            if self.data_in_mem:
                for view in self.allowed_views:
                    df = self.data_frame[self.data_frame['us_id'] == exam].reset_index(drop=True)
                    all_exam_indx = df[df['view'] == view].index
                    if len(all_exam_indx) > 0:
                        rndm_indx = random.choice(all_exam_indx)
                        row = self.data_frame.iloc[rndm_indx]
                        img = row['img']
                        img = self.data_aug_img.transform_values(img).transpose(3, 0, 1, 2)
                        data_list.append(img)
                        flow = row['flow']
                        flow = self.data_aug_flow.transform_values(flow).transpose(3, 0, 1, 2)
                        data_list.append(flow)
                    else:
                        img, flow = self.generate_dummy_data()
                        data_list.append(img)
                        data_list.append(flow)
            else:
                for view in self.allowed_views:
                    df = self.targets[self.targets['us_id'] == exam].reset_index(drop=True)
                    all_exam_indx = df[df['view'] == view].index
                    if len(all_exam_indx) > 0:
                        rndm_indx = random.choice(all_exam_indx)
                        img, flow, _, _, _, _ = self.read_image_data(tuple(df.iloc[rndm_indx]))
                        img = self.data_aug_img.transform_values(img).transpose(3, 0, 1, 2)
                        data_list.append(img)
                        flow = self.data_aug_flow.transform_values(flow).transpose(3, 0, 1, 2)
                        data_list.append(flow)
                    else:
                        img, flow = self.generate_dummy_data()
                        data_list.append(img)
                        data_list.append(flow)

        if not self.only_use_complete:
            # Replace dummy views with real views.
            dummy_indexes = []
            real_indexes = []
            for i in range(0, len(self.allowed_views)*2, 2):
                if not data_list[i].all():
                    dummy_indexes.append(i)
                else:
                    real_indexes.append(i)
            if len(dummy_indexes):
                # Sort real views by priority specified by Eva
                real_indexes.sort(reverse=True)
                if real_indexes[0] == 6:
                    real_indexes.append(real_indexes.pop(0))
                for i in dummy_indexes:
                    # Replace dummy image data with top prio real image data
                    data_list[i] = data_list[real_indexes[0]]
                    # Replace dummy flow data with top prio real flow data
                    data_list[i + 1] = data_list[real_indexes[0] + 1]

        return data_list, np.expand_dims(target, axis=0).astype(np.float32), index, exam, iid_list

    def load_data_into_mem(self):
        nprocs = mp.cpu_count()
        print(f"Number of CPU cores: {nprocs}")
        pool = mp.Pool(processes=nprocs-2)
        iterator = self.targets.itertuples(index=False, name=None)
        result = pool.map(self.read_image_data, iterator)
        pool.close()
        pool.join()
        data_list = []
        for r in result:
            data_list.append(r)
        self.data_frame = pd.DataFrame(data_list, columns=['img', 'flow', 'target', 'us_id', 'iid', 'view'])
        print('All data loaded into memory')

    def load_data_to_disk(self):
        nprocs = mp.cpu_count()
        print(f"Number of CPU cores: {nprocs}")
        pool = mp.Pool(processes=nprocs-2)
        iterator = self.targets.itertuples(index=False, name=None)
        pool.map(self.write_data_to_disk, iterator)
        pool.close()
        pool.join()
        print('All data processed and loaded to disk')

    def write_data_to_disk(self, data):
        uid, _, _, _, _, file_img, file_flow, target, rwaves = data
        if self.rwave_only:
            if isinstance(pd.eval(rwaves), float):
                return 0
        folder_img = os.path.join(self.temp_folder_img, uid)
        fp_img = os.path.join(folder_img, file_img)
        if not os.path.exists(folder_img):
            os.makedirs(folder_img)
        folder_flow = os.path.join(self.temp_folder_flow, uid)
        fp_flow = os.path.join(folder_flow, file_flow)
        if not os.path.exists(folder_flow):
            os.makedirs(folder_flow)
        if os.path.exists(fp_img) and (not os.path.getsize(fp_img) == 0) and \
                os.path.exists(fp_flow) and (not os.path.getsize(fp_flow) == 0):
            return 0
        else:
            img, flow, _, _, _, _ = self.read_image_data(data)
            np.save(fp_img, img)
            np.save(fp_flow, flow)
        return 0

    def read_image_data(self, data):
        uid, iid, view, fps, hr, file_img, file_flow, target, rwaves = data
        fp_img = os.path.join(os.path.join(self.base_folder_img, uid), file_img)
        fp_flow = os.path.join(os.path.join(self.base_folder_flow, uid), file_flow)
        try:
            if target is None:
                raise ValueError("Target is None")
            # Process img
            img = np.load(fp_img, allow_pickle=True)
            if img is None:
                raise ValueError("Img is None")
            # Process flow
            flow = np.load(fp_flow, allow_pickle=True)
            if flow is None:
                raise ValueError("Flow is None")
            if not self.preprocessed_data_on_disk:
                img = self.data_aug_img.transform_size(img, fps, hr, pd.eval(rwaves))
                flow = self.data_aug_flow.transform_size(flow, fps, hr, pd.eval(rwaves))
            return img, flow, target, uid, iid, view
        except Exception as e:
            print("Failed to get item for img file: {} and flow file: {} with exception: {}".format(file_img, file_flow, e))

    def filter_incomplete_exams(self):
        '''
        Can be used to filter targets for all exams that does not have at least one of every view in allowed_views.
        Not currently used.
        :return: Pandas dataframe with only exams that have one of every view.
        '''
        unique_exams = self.targets.drop_duplicates('us_id')['us_id'].copy()
        view_arr = []
        for row in unique_exams:
            # Only include exams which contain 4-chamber view
            if self.is_eval_set and self.minimum_validation:
                view_arr.append(all(elem in self.targets[self.targets['us_id'] == row]['view'].values for elem in [4]))
            else:
                view_arr.append(all(elem in self.targets[self.targets['us_id'] == row]['view'].values for elem in self.allowed_views))
        filtered_ue = unique_exams[view_arr].copy()
        filtered_targets = self.targets[self.targets['us_id'].isin(filtered_ue)].copy().reset_index(drop=True)
        self.targets = filtered_targets
        self.unique_exams = self.targets.drop_duplicates('us_id')[['us_id', 'target']]
        if self.is_eval_set and self.minimum_validation:
            print("Unique validation exams:", len(self.unique_exams))

    def combinate(self, items, size=4):
        '''
        Returns a generator that yields pairs of every combination of items in all categories
        :param items: Dictionary with the format {category A: list of items in category A, category B: list of items in category B}
        :param size: how many items we return in each combination.
        :return: A generator that yields every combination.
        '''
        for cats in itertools.combinations(items.keys(), size):
            cat_items = [[products for products in items[cat]] for cat in cats]
            for x in itertools.product(*cat_items):
                yield zip(cats, x)

    def generate_all_combinations(self):
        '''
        Generates all possible combinations of views in an examination and uses all those combinations as the dataset.
        If a view is missing, the examination is still used and the view is set to None.
        As an example, if a unique exam has one view of type A, two views of type B and 2 views of type C then four
        different combinations would be added to the final dataset.
        :return:
        '''
        all_generated_combinations = []
        for ue in self.unique_exams.itertuples():
            new_dict = {}
            t = self.targets[self.targets['us_id'] == ue.us_id]
            for view in self.allowed_views:
                ue_data = t[t.view == view][['instance_id', 'fps', 'hr', 'filename_img', 'filename_flow', 'rwaves']].values.tolist()
                if len(ue_data) > 0:
                    new_dict[view] = ue_data
                else:
                    new_dict[view] = [[None, None, None, None, None, None]]
            ue_combs = self.combinate(new_dict, len(self.allowed_views))

            for uec in ue_combs:
                pd_dict = {'us_id': ue.us_id, 'target': ue.target}
                for view, data in uec:
                    pd_dict['instance_id_' + str(view)] = data[0]
                    pd_dict['fps_' + str(view)] = data[1]
                    pd_dict['hr_' + str(view)] = data[2]
                    pd_dict['filename_img_' + str(view)] = data[3]
                    pd_dict['filename_flow_' + str(view)] = data[4]
                    pd_dict['rwaves_' + str(view)] = data[5]
                all_generated_combinations.append(pd_dict)
        return pd.DataFrame(all_generated_combinations)

    def generate_dummy_data(self):
        img = np.zeros((1, self.data_aug_img.transforms.target_length,
                        self.data_aug_img.transforms.target_height,
                        self.data_aug_img.transforms.target_width), dtype=np.float32)
        flow = np.zeros((2, self.data_aug_flow.transforms.target_length,
                         self.data_aug_flow.transforms.target_height,
                         self.data_aug_flow.transforms.target_width), dtype=np.float32)
        return img, flow


class MultiStreamDatasetNoFlow(MultiStreamDataset):
    def __init__(self, cfg_data, cfg_transforms, cfg_augmentations, target_file, is_eval_set):
        super().__init__(cfg_data, cfg_transforms, cfg_augmentations, target_file, is_eval_set)

    def __getitem__(self, index):
        data_list = []
        iid_list = [] 
        if self.is_eval_set:
            if self.data_in_mem:
                df = self.targets_combinations.iloc[index]
                exam = df['us_id']
                target = df['target']
                for view in self.allowed_views:
                    iid = df['instance_id_' + str(view)]
                    if np.isnan(iid):
                        img, _ = self.generate_dummy_data()
                        data_list.append(img)
                    else:
                        row = self.data_frame[(self.data_frame['us_id'] == exam) & (self.data_frame['view'] == view) & (self.data_frame['iid'] == iid)]
                        img = row['img']
                        img = self.data_aug_img.transform_values(img).transpose(3, 0, 1, 2)
                        data_list.append(img)
            else:
                df = self.targets_combinations.iloc[index]
                exam = df['us_id']
                target = df['target']
                for view in self.allowed_views:
                    iid = df['instance_id_' + str(view)]
                    if np.isnan(iid):
                        img, _ = self.generate_dummy_data()
                        data_list.append(img)
                    else:
                        fps = df['fps_' + str(view)]
                        hr = df['hr_' + str(view)]
                        file_img = df['filename_img_' + str(view)]
                        file_flow = df['filename_flow_' + str(view)]
                        if self.rwave_only:
                            rwaves = df['rwaves_' + str(view)]
                        else:
                            rwaves = None
                        img, _, _, _, _, _ = self.read_image_data(tuple((exam, 0, 0, fps, hr, file_img, file_flow, target, rwaves)))
                        img = self.data_aug_img.transform_values(img).transpose(3, 0, 1, 2)
                        data_list.append(img)
        else:
            exam = self.unique_exams.iloc[index].us_id
            target = self.unique_exams.iloc[index].target
            if self.data_in_mem:
                for view in self.allowed_views:
                    df = self.data_frame[self.data_frame['us_id'] == exam].reset_index(drop=True)
                    all_exam_indx = df[df['view'] == view].index
                    if len(all_exam_indx) > 0:
                        rndm_indx = random.choice(all_exam_indx)
                        row = self.data_frame.iloc[rndm_indx]
                        img = row['img']
                        img = self.data_aug_img.transform_values(img).transpose(3, 0, 1, 2)
                        data_list.append(img)
                    else:
                        img, _ = self.generate_dummy_data()
                        data_list.append(img)
            else:
                for view in self.allowed_views:
                    df = self.targets[self.targets['us_id'] == exam].reset_index(drop=True)
                    all_exam_indx = df[df['view'] == view].index
                    if len(all_exam_indx) > 0:
                        rndm_indx = random.choice(all_exam_indx)
                        img, _, _, _, _, _ = self.read_image_data(tuple(df.iloc[rndm_indx]))
                        img = self.data_aug_img.transform_values(img).transpose(3, 0, 1, 2)
                        data_list.append(img)
                    else:
                        img, _ = self.generate_dummy_data()
                        data_list.append(img)
        if not self.only_use_complete:
            # Replace dummy views with real views.
            dummy_indexes = []
            real_indexes = []
            for i in range(0, len(self.allowed_views)):
                if not data_list[i].all():
                    dummy_indexes.append(i)
                else:
                    real_indexes.append(i)
            if len(dummy_indexes):
                # Sort real views by priority specified by Eva
                real_indexes.sort(reverse=True)
                if real_indexes[0] == 3:
                    real_indexes.append(real_indexes.pop(0))
                for i in dummy_indexes:
                    # Replace dummy image data with top prio real image data
                    data_list[i] = data_list[real_indexes[0]]

        return data_list, np.expand_dims(target, axis=0).astype(np.float32), index, exam, iid_list
