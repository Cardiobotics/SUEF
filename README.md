# SUEF

Training and validation for multi-stream models using echocardiographic data.
Implemented models are I3D, I3D+BERT and ResNext3D.

Usage:

Run main.py
Set default config using the yaml files in the cfg folder.
Temporary changes can be added from command line, for example: "python main.py performance.parallel_model=False augmentations.train_a.gaussian_noise=True".
