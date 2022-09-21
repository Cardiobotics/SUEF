import torch
import pandas as pd
import numpy as np
from data import data_augmentations
import os
import multiprocessing as mp
import random
import itertools
import time


class EchoMultiStreamDataset(torch.utils.data.Dataset):
    def __init__(self, cfg_data, cfg_transforms, cfg_augmentations, target_file, is_eval_set):
        super(EchoMultiStreamDataset).__init__()
        assert not (cfg_data.data_in_mem and cfg_data.preprocessed_data_on_disk)

        self.is_eval_set = is_eval_set
        if self.is_eval_set:
            self.split = "VAL"
            self.split_path = "val_echonet_data/"
        else:
            self.split = "TRAIN" 
            self.split_path = "train_echonet_data/"
        self.preprocessed_data_on_disk = cfg_data.preprocessed_data_on_disk
        if cfg_transforms.scale_output:
            self.targets['target'] = self.targets['target'].apply(lambda x: x / 100)
        self.data_aug_img = data_augmentations.DataAugmentations(cfg_transforms, cfg_augmentations.img)
        self.data_aug_flow = data_augmentations.DataAugmentations(cfg_transforms, cfg_augmentations.flow)
        root = "/media/richard/4fe0f1b8-1f60-4a27-a7df-72f752f56fa5/echo-evaluation/EchoNet-Dynamic/"
        self.base_folder_img = os.path.join(root,os.path.join(self.split_path, "echo_data_" + self.split))
        self.base_folder_flow = os.path.join(root,os.path.join(self.split_path, "echo_data_flow_" + self.split))
        self.only_use_complete = cfg_data.only_use_complete_exams
        self.df = pd.read_csv(os.path.abspath(target_file), sep=cfg_data.file_sep)
        self.data_list = self.df.loc[(self.df['Split'] == self.split), ["FileName", "FPS", "EF"]].to_numpy()
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_list = []
        iid_list = []
        if self.is_eval_set:
            file_name, fps, target = self.data_list[index]
            img, flow, _ = self.read_echo_data((file_name, fps, target))
            img = self.data_aug_img.transform_values(img).transpose(3, 0, 1, 2)
            data_list.append(img)
            flow = self.data_aug_flow.transform_values(flow).transpose(3, 0, 1, 2)
            data_list.append(flow)
        else:
            file_name, fps, target = self.data_list[index]
            img, flow, _ = self.read_echo_data((file_name, fps, target))
            img = self.data_aug_img.transform_values(img).transpose(3, 0, 1, 2)
            data_list.append(img)
            flow = self.data_aug_flow.transform_values(flow).transpose(3, 0, 1, 2)
            data_list.append(flow)


        return data_list, np.expand_dims(target, axis=0).astype(np.float32), index, file_name 
    
    def read_echo_data(self, data):
        file_name, fps, target = data
        fp_img = os.path.join(self.base_folder_img, file_name + ".npy")
        fp_flow = os.path.join(self.base_folder_flow, file_name + "_flow_.npy")
        img = np.load(fp_img, allow_pickle=True)
        flow = np.load(fp_flow, allow_pickle=True)
        
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
                img = self.data_aug_img.transform_size(img, fps, None, False)
                flow = self.data_aug_flow.transform_size(flow, fps, None, False)
            return img, flow, target 
        except Exception as e:
            print("Failed to get item for img file: {} and flow file: {} with exception: {}".format(fp_img, fp_flow, e))
        
