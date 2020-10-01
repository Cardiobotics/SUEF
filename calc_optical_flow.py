import cv2
import numpy as np
import time
import argparse
from utils import load_filenames
import os
import multiprocessing as mp


def generate_flow(file):
    file_path, file_name = file
    uid = file_name[0:13]
    out_file_name = file_name[0:-4] + 'flow_.npy'
    out_folder = os.path.join(args.output_folder, uid)
    out_path = os.path.join(out_folder, out_file_name)

    if not os.path.isfile(out_path):
        time_start = time.time()
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
           
        data = np.load(file_path, allow_pickle=True)
        data = np.squeeze(data)

        prvs = data[0]
        flows = np.zeros((len(data) - 1, data.shape[1], data.shape[2], 2))

        for i in range(len(data) - 1):
            nxt = data[i + 1]
            tv = cv2.optflow.createOptFlow_DualTVL1()
            tv.setEpsilon(0.05)
            tv.setWarpingsNumber(1)
            tv.setScalesNumber(1)
            tv_flow = tv.calc(prvs, nxt, None)
            flows[i] = tv_flow
            prvs = nxt
        flows = np.round(flows).astype(np.int8)

        np.save(out_path, flows)
        time_delta = time.time() - time_start
        #print("The process: {} took {} seconds to finish calculating flow".format(mp.current_process(), time_delta))
        return 1
    else:
        return 0


parser = argparse.ArgumentParser(description='Converts input numpy grayscale data into optical flow')
parser.add_argument('--input_folder', default='/media/ola/7540de01-b8d5-4df4-883c-1a8429f18b56/img/', type=str,
                    help='Input folder containing videos in 3-dimensional npy format')
parser.add_argument('--output_folder', default='/media/ola/324ac400-018f-4d6b-941e-361b54f3e5f6/flow/',
                    type=str, help='Output folder where we save the resulting optical flow')

args = parser.parse_args()

sub_folders = next(os.walk(args.input_folder))[1]
files = []
print(len(sub_folders))
for sf in sub_folders:
    files = files + load_filenames(os.path.join(args.input_folder, sf))
print(len(files))
time_start = time.time()

nprocs = mp.cpu_count()
print(f"Number of CPU cores: {nprocs}")
pool = mp.Pool(processes=nprocs)
result = pool.imap_unordered(generate_flow, files, chunksize=len(files) // nprocs)
print(sum(result))
print('Flow generated')
pool.close()
pool.join()
time_flow = time.time() - time_start
print("Time: {}".format(time_flow))