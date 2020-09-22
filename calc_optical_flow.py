import cv2
import numpy as np
import time
import argparse
from utils import load_filenames
import os
import multiprocessing as mp


def generate_flow(file):
    file_path, file_name = file
    data = np.load(file_path, allow_pickle=True)
    data = np.squeeze(data)

    prvs = data[0]

    flows = np.zeros((len(data) - 1, data.shape[1], data.shape[2], 2))

    for i in range(len(data) - 1):
        nxt = data[i + 1]
        flow = cv2.calcOpticalFlowFarneback(prvs, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flows[i] = flow
    flows = flows.astype(np.int8)
    out_file_name = file_name[0:-4] + '_flow_.npy'
    np.save(os.path.join(args.output_folder, out_file_name), flows)
    return 1


parser = argparse.ArgumentParser(description='Converts input numpy grayscale data into optical flow')
parser.add_argument('--input_folder', default='/media/ola/324ac400-018f-4d6b-941e-361b54f3e5f6/img/2', type=str,
                    help='Input folder containing videos in 3-dimensional npy format')
parser.add_argument('--output_folder', default='/media/ola/324ac400-018f-4d6b-941e-361b54f3e5f6/flow/2',
                    type=str, help='Output folder where we save the resulting optical flow')

args = parser.parse_args()

files = load_filenames(args.input_folder)
if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

nprocs = mp.cpu_count()
print(f"Number of CPU cores: {nprocs}")
pool = mp.Pool(processes=nprocs)
result = pool.map(generate_flow, files)
print('Flow generated')
pool.close()
pool.join()
