import argparse
import cnn_model

def main():

    parser = argparse.ArgumentParser('Train a CNN model')
    parser.add_argument('--traindata', help='The folder containing'
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