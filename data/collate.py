import torch
import numpy as np


class CustomCollate():
    def __init__(self, padding_enabled):
        self.padding_enabled = padding_enabled

    def collate_fn(self, input_list):
        '''
        Custom collate method for a torch.utils.DataLoader object that pads the size of each video in the batch
        to the maximum size for a video in the batch.
        :param input_list: The batch of videos and targets, sent from DataLoader
        :return: The processed batch of videos and targets.
        '''
        batch_size = len(input_list)
        channels = max([i.shape[0] for i in input_list])
        max_length = max([i.shape[1] for i in input_list])
        max_height = max([i.shape[2] for i in input_list])
        max_width = max([i.shape[3] for i in input_list])

        img_tensor = torch.zeros((batch_size, channels, max_length, max_height, max_width), dtype=float)
        target_tensor = torch.zeros(batch_size, dtype=float)

        for j, (img, target) in input_list:
            if self.padding_enabled:
                pad_sequence = ((0, 0), (0, 0), (0, 0), (0, 0))
                if img.shape[1] < max_length:
                    pad_sequence[0] = (0, max_length - img.shape[1])
                if img.shape[2] < max_height:
                    pad_sequence[1] = ((max_height - img.shape[2])/2, (max_height - img.shape[2])/2)
                if img.shape[3] < max_width:
                    pad_sequence[2] = ((max_width - img.shape[3]) / 2, (max_width - img.shape[3]) / 2)
                if not pad_sequence == ((0, 0), (0, 0), (0, 0), (0, 0)):
                    img = np.pad(img, pad_width=pad_sequence)

            img_tensor[j] = torch.tensor(img)
            target_tensor[j] = torch.tensor(target)

        return img_tensor, target_tensor
