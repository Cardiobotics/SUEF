#!/bin/bash

/home/ola/anaconda3/envs/david/bin/python /home/ola/Projects/SUEF/main.py augmentations.train_a.gaussian_noise=True
/home/ola/anaconda3/envs/david/bin/python /home/ola/Projects/SUEF/main.py augmentations.train_a.speckle=True
/home/ola/anaconda3/envs/david/bin/python /home/ola/Projects/SUEF/main.py augmentations.train_a.salt_and_pepper=True
/home/ola/anaconda3/envs/david/bin/python /home/ola/Projects/SUEF/main.py augmentations.train_a.translate_v=True augmentations.train_a.translate_h=True
/home/ola/anaconda3/envs/david/bin/python /home/ola/Projects/SUEF/main.py augmentations.train_a.rotate=True
/home/ola/anaconda3/envs/david/bin/python /home/ola/Projects/SUEF/main.py augmentations.train_a.local_blackout=True
/home/ola/anaconda3/envs/david/bin/python /home/ola/Projects/SUEF/main.py augmentations.train_a.local_intensity=True

