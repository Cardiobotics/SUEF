import pandas as pd
import numpy as np
import argparse
import os
from arguments import add_transform_config_args
from data import data_augmentations
import multiprocessing as mp


def main():

    parser = argparse.ArgumentParser()
    parser = add_transform_config_args(parser)
    parser.add_argument('--target_file', type=str, help='Target file with files that should be processed')
    parser.add_argument('--input_folder_img', type=str, help='Folder where original img (numpy) files are located')
    parser.add_argument('--input_folder_flow', type=str, help='Folder where original flow (numpy) files are located')
    parser.add_argument('--output_folder_img', type=str, help='Folder where transformed img (numpy) files should be saved')
    parser.add_argument('--output_folder_flow', type=str, help='Folder where transformed flow (numpy) files should be saved')

    args = parser.parse_args()

    global transforms

    transforms = data_augmentations.DataAugmentations(args, False)

    targets = pd.read_csv(os.path.abspath(args.target_file), sep=';')

    global input_folder_img
    global input_folder_flow
    global output_folder_img
    global output_folder_flow

    input_folder_img = args.input_folder_img
    input_folder_flow = args.input_folder_flow
    output_folder_img = args.output_folder_img
    output_folder_flow = args.output_folder_flow

    load_data_to_disk(targets)


def load_data_to_disk(targets):
    nprocs = mp.cpu_count()
    print(f"Number of CPU cores: {nprocs}")
    pool = mp.Pool(processes=nprocs-1)
    iterator = targets.itertuples(index=False, name=None)
    pool.map(write_data_to_disk, iterator)
    pool.close()
    pool.join()
    print('All data processed and loaded to disk')


def write_data_to_disk(data):
    uid, _, _, _, _, file_img, file_flow, target = data
    folder_img = os.path.join(output_folder_img, uid)
    fp_img = os.path.join(folder_img, file_img)
    if not os.path.exists(folder_img):
        os.makedirs(folder_img)
    folder_flow = os.path.join(output_folder_flow, uid)
    fp_flow = os.path.join(folder_flow, file_flow)
    if not os.path.exists(folder_flow):
        os.makedirs(folder_flow)
    if os.path.exists(fp_img) and (not os.path.getsize(fp_img) == 0) and \
            os.path.exists(fp_flow) and (not os.path.getsize(fp_flow) == 0):
        return 0
    else:
        img, flow, _, _, _, _ = read_image_data(data)
        np.save(fp_img, img)
        np.save(fp_flow, flow)
    return 0


def read_image_data(data):
    uid, iid, view, fps, hr, file_img, file_flow, target = data
    fp_img = os.path.join(os.path.join(input_folder_img, uid), file_img)
    fp_flow = os.path.join(os.path.join(input_folder_flow, uid), file_flow)
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
        img = transforms.transform_size(img, fps, hr)
        flow = transforms.transform_size(flow, fps, hr)
        return img, flow, target, uid, iid, view
    except Exception as e:
        print("Failed to get item for img file: {} and flow file: {} with exception: {}".format(file_img, file_flow, e))


if __name__ == '__main__':
    main()