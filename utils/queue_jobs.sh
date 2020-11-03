#!/bin/bash

/home/ola/anaconda3/envs/david/bin/python /home/ola/Projects/SUEF/main.py augmentations.train_a.translate_v=True augmentations.train_a.translate_h=True
/home/ola/anaconda3/envs/david/bin/python /home/ola/Projects/SUEF/main.py augmentations.train_a.salt_and_pepper=True
/home/ola/anaconda3/envs/david/bin/python /home/ola/Projects/SUEF/main.py augmentations.train_a.speckle=True
/home/ola/anaconda3/envs/david/bin/python /home/ola/Projects/SUEF/main.py augmentations.train_a.gaussian_noise=True
/home/ola/anaconda3/envs/david/bin/python /home/ola/Projects/SUEF/main.py transforms.train_t.rescale_fps=True transforms.train_t.rescale_fphb=False transforms.eval_t.rescale_fps=True transforms.eval_t.rescale_fphb=False