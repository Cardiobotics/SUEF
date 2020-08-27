# Input: Dicom folder name (lista?), Fil med viewlabels
# Läs in Dicom baserat på labels
# Normalisera och beskär bilder

import argparse
import pydicom
import numpy as np
import pandas as pd
import pp_transforms
import os

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
print(args.dfolder)
print(args.vlabel)
print(args.views)
print(args.vlabel_sep)

pd_vl = pd.read_csv(args.vlabel, sep=args.vlabel_sep)


def load_filenames(path):
    files = []
    for dirName, _, fileList in os.walk(path):
        for filename in fileList:
            files.append(os.path.join(dirName, filename))
    return files


for p in args.dfolder:
    files = load_filenames(p)
    for f in files:
        data = pydicom.read_file(f)
        u_id = data.PatientID
        inst_id = data.InstanceNumber
        v_row = pd_vl.loc[(pd_vl['us_id'] == u_id) & (pd_vl['instance_id'] == inst_id)]
        if len(v_row) > 1:
            raise ValueError("More than 1 matching tuple of user id and instance id in view label file")
        if len(v_row) < 1:
            continue
        if not v_row['prediction'].item() in args.views:
            continue
        else:
            flags = {'gs':True}
            img = pp_transforms.apply_transforms(data, flags)
            print(img.shape)
            out_folder = os.path.join(args.obfolder, str(v_row['prediction'].item()))
            out_file_name = '{}_{}.npy'.format(u_id, str(inst_id).zfill(2))
            out_file_path = os.path.join(out_folder, out_file_name)


