
train_transforms = {
    'grayscale': True,
    'normalize_input': True,
    'scale_output': True,
    'noise': True,
    'speckle': True,
    'rescale_fps': True,
    'resize_frames': True,
    'crop': 3,
    'pad': 2,
    'org_height': 484,
    'org_width': 636,
    'target_height': int(0.20*484),
    'target_width': int(0.20*636),
    'target_length': 30,
    'target_fps': 25
}

val_transforms = {
    'grayscale': True,
    'normalize_input': True,
    'scale_output': True,
    'noise': False,
    'speckle': False,
    'rescale_fps': True,
    'resize_frames': True,
    'crop': 3,
    'pad': 2,
    'org_height': 484,
    'org_width': 636,
    'target_height': int(0.20*484),
    'target_width': int(0.20*636),
    'target_length': 30,
    'target_fps': 25
}

allowed_views = [0, 2, 4, 12, 20]

resnext_settings = {
    'model_depth': 50,
    'cardinality': 32,
    'n_classes': 1,
    'n_input_channels': 1,
    'shortcut_type': 'B',
    'conv1_t_size': 7,
    'conv1_t_stride': 1
}