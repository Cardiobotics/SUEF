# Input: Dicom folder name (lista?), Fil med viewlabels
# Läs in Dicom baserat på labels
# Normalisera och beskär bilder

import argparse
import pydicom
import numpy as np
import pandas as pd
import pp_transforms
import os
from tqdm import tqdm

def main():

    parser = argparse.ArgumentParser('Reads dicom files and view labels and processes into npy')
    parser.add_argument('--dfolder', action='append',
                        help='The root folder containing Dicom files. Use this argument '
                             'multiple times if you have multiple source folders')
    parser.add_argument('--vlabel', help='The path to the csv file containing view labels.')
    parser.add_argument('--views', type=int, action='append', help='The view to process. Use this argument multiple times'
                                                                   'if you have multiple views you want to process')
    parser.add_argument('--vlabel_sep', default=';', help='The separator for the view label csv file. Default is ;')
    parser.add_argument('--obfolder', help='Output base folder. Folder structure will be generated with this as root')

    args = parser.parse_args()

    pd_vl = load_viewlabels(args.vlabel, args.vlabel_sep)

    t_flags = {'gs':True}

    load_and_process_dicom(pd_vl, args.dfolder, args.views, args.obfolder, t_flags)


def load_viewlabels(vlabel_path, vlabel_sep):
    '''
    Loads a csv file into a pandas dataframe.
    :param vlabel_path: Path to csv file. (Str)
    :param vlabel_sep: Separator for csv. (Str)
    :return: pandas dataframe. (Pandas DataFrame)
    '''
    pddf = pd.read_csv(vlabel_path, sep=vlabel_sep)
    return pddf


def load_filenames(path):
    '''
    Return all filenames that exist in the folder and subfolders specified by path as a list.
    :param path: Path to base folder (Str)
    :return: List of full paths to all files (List of Str)
    '''
    files = []
    for dirName, _, fileList in os.walk(path):
        for filename in fileList:
            files.append(os.path.join(dirName, filename))
    return files

def load_and_process_dicom(pd_viewlabels, dicom_folder, accepted_views, output_base_folder, transform_flags):
    '''
    Loads all dicom files and checks if they are the accepted view.
    If they are, applies all flagged transforms to image data and saves it to disk as a npy file.
    :param pd_viewlabels: Viewlabels dataframe (Pandas Dataframe)
    :param dicom_folder: Paths to dicom files (List of Str)
    :param accepted_views: Accepted view angles, specified by number (List of Int)
    :param output_base_folder: Path to the output base folder (Str)
    :param transform_flags: Flags for all transforms found in pp_transforms (Dict)
    :return:
    '''
    for p in dicom_folder:
        files = load_filenames(p)
        for f in files:
            data = pydicom.read_file(f)
            u_id = data.PatientID
            inst_id = data.InstanceNumber
            v_row = pd_viewlabels.loc[(pd_viewlabels['us_id'] == u_id) & (pd_viewlabels['instance_id'] == inst_id)]
            if len(v_row) > 1:
                raise ValueError("More than 1 matching tuple of user id and instance id in view label file")
            if len(v_row) < 1:
                continue
            if not v_row['prediction'].item() in accepted_views:
                continue
            else:
                img = pp_transforms.apply_transforms(data, transform_flags)
                print(img.shape)
                out_folder = os.path.join(output_base_folder, str(v_row['prediction'].item()))
                out_file_name = '{}_{}.npy'.format(u_id, str(inst_id).zfill(2))
                out_file_path = os.path.join(out_folder, out_file_name)


if __name__ == "__main__":
    main()