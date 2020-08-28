import torch
import numpy as np
import pandas as pd
import os


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, target_file, target_file_sep, uid_len):
        super(CustomDataset).__init__()

        self.targets = pd.read_csv(os.path.abspath(target_file), sep=target_file_sep)
        self.img_dict = self.load_filenames( os.path.abspath(image_path), uid_len)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        uid = self.targets.iloc[index]['us_id']
        target = self.targets.iloc[index]['target']
        f_path = self.img_dict[uid]
        print(f_path)
        img = np.load(os.path.abspath(f_path), allow_pickle=True)
        return img, target


    def load_filenames(self, path, uid_len):
        '''
        Returns a dictionary of user_id:filepath
        It is assumed that the first uid_len characters in the filename is the user_id
        :param path: root directory containing all files (Str)
        :param uid_len: length of user_id in the filename (Int)
        :return: dictionary of userids and paths (Dict)
        '''
        files = {}
        for dirName, _, fileList in os.walk(path):
            for filename in fileList:
                files[filename[0:uid_len]] = os.path.join(dirName, filename)
        return files