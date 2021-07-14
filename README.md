# Install

* Install [CUDA 10.1] (https://developer.nvidia.com/cuda-tookit) and [cuDNN 8.0](https://developer.nvidia.com/cudnn)

* Install system dependencies: sudo apt install python3-pip

* Install all dependencies: pip3 install -r requirements.txt

## Training

Run ``python train_ddp.py`` to start training using the config set in cfg/config.yaml.  
Configurations parameters can be overridden with arguments from command-line.  
For example: ``python3 train_ddp.py optimizer.learning_rate=0.01 data.data_in_mem=False``
Please note that config.yaml reads several parameters from different sub-files which also can (and should) be updated. 

## Evaluation

Run ``python evaluate_model.py`` to start evaluation using the config set in cfg/config.yaml.
Not all config options are used during evaluation (for example data_loader.drop_last is always set to False).
Results are saved in a csv file to disk.
