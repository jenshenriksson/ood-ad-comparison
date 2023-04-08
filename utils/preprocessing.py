import segmentation_models_pytorch as smp

import albumentations as albu 
from albumentations.augmentations.geometric.resize import Resize

def get_preprocessing_fn(): 
    ENCODER = 'mobilenet_v2'
    ENCODER_WEIGHTS = 'imagenet'
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    return preprocessing_fn


def get_validation_augmentation(height=1024, width=2048):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.CenterCrop(height=height, width=width, always_apply=True),
    ]
    return albu.Compose(test_transform)


def get_visualization_augmentation(height=376, width=752): 
    transform = [
        albu.SmallestMaxSize(max_size=376), 
        # albu.RandomCrop(width=width, height=height) 
    ]
    return albu.Compose(transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def pspnet_augmentation(psp_width=2041, psp_height=1017):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.CenterCrop(height=psp_height, width=psp_width, always_apply=True),
    ]
    return albu.Compose(test_transform)
    