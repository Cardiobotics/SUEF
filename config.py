
train_transforms = {
    'grayscale': True,
    'float': True,
    'noise': False,
    'speckle': False,
    'rescale_fps': True,
    'resize_frames': False,
    'crop': 3,
    'target_height': 300,
    'target_width': 300,
    'target_length': 10,
    'target_fps': 12
}

val_transforms = {
    'grayscale': True,
    'float': True,
    'noise': False,
    'speckle': False,
    'rescale_fps': True,
    'resize_frames': False,
    'crop': 3,
    'target_height': 300,
    'target_width': 300,
    'target_length': 10,
    'target_fps': 12
}

allowed_views = [0, 2, 4, 20]