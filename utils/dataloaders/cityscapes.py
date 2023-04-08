import cv2 
import numpy as np
from torch.utils.data import Dataset as Dataset
import os 

import sys
sys.path.append('..')
import time

import torch 
from torch.utils.data import DataLoader

from tqdm.notebook import tqdm
from utils.utils import list_dir_recursive
import albumentations as albu
from albumentations.augmentations.geometric.resize import Resize


HEIGHT = 384
WIDTH = 768

albuHeight = albu.Compose([
    albu.LongestMaxSize(max_size=WIDTH, always_apply=True), 
    albu.CenterCrop(height=HEIGHT, width=WIDTH, always_apply=True),
])
albuWidth = albu.Compose([
    albu.SmallestMaxSize(max_size=HEIGHT, always_apply=True),
    albu.CenterCrop(height=HEIGHT, width=WIDTH, always_apply=True),
])




class CityScapes(Dataset):
    """
    CityScapes Dataset. Read images, apply augmentation and preprocessing transformations.
    DataLoader modified from https://github.com/qubvel/segmentation_models.pytorch 
    
    Information regarding CLASSES available here: https://www.cityscapes-dataset.com/dataset-overview/
    
    Args:
        images (list): List of all paths to the images for the dataset 
        labels (list): List of all paths to the labels for the dataset 
        class_values (list): A selection of classes to extract from segmentation mask if not all is to be used. 
            (e.g. ['car'] or ['car', 'person'], etc.) 
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = [
        'unlabeled', 
        'ego vehicle',
        'rectification border', 
        'out of roi', 
        'static', 
        'dynamic', 
        'ground', 
        'road', 
        'sidewalk', 
        'parking', 
        'rail track',  
        'building', 
        'wall', 
        'fence', 
        'guard rail', 
        'bridge', 
        'tunnel',  
        'pole', 
        'polegroup', 
        'traffic light', 
        'traffic sign',  
        'vegetation', 
        'terrain',  
        'sky',  
        'person', 
        'rider',  
        'car', 
        'truck', 
        'bus', 
        'caravan',
        'trailer', 
        'train', 
        'motorcycle', 
        'bicycle',   
        "garage",                  # Added for KITTI-360 (Used to be Licence Plate for CityScapes)
        "gate",                    # Added for KITTI-360
        "stop",                    # Added for KITTI-360
        "smallpole",               # Added for KITTI-360
        "lamp",                    # Added for KITTI-360
        "trash bin",               # Added for KITTI-360
        "vending machine",         # Added for KITTI-360
        "box",                     # Added for KITTI-360
        "unknown construction",    # Added for KITTI-360 
        "unknown vehicle",         # Added for KITTI-360
        "unknown object",          # Added for KITTI-360
        "license plate"            # Added for KITTI-360
    ]
    
    def __init__(
            self, 
            images, 
            labels, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
            preload=False, 
    ):
        self.images_fps = images
        self.masks_fps = labels
        self.preload = preload 
        
        # convert str names to class values on masks
        if not classes: 
            classes = self.CLASSES
       
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.augmentation = augmentation
        self.preprocessing = preprocessing
                
        # Store images before dataloader. 
        if self.preload: 
            self.preloadedImages = []
            self.preloadedMasks = [] 
            
            for i in tqdm(range(len(self.images_fps))):
                image = cv2.imread(self.images_fps[i])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mask = cv2.imread(self.masks_fps[i], 0)
                
                # Resize
                if image.shape[1] / image.shape[0] >= 2: 
                    image = albuWidth(image=image)['image']
                    mask = albuWidth(image=mask)['image']
                else: 
                    image = albuHeight(image=image)['image']
                    mask = albuHeight(image=mask)['image']
                
                self.preloadedImages.append(image)
                self.preloadedMasks.append(mask) 
    
    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype(np.float32)
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask
        
    def __len__(self):
        return len(self.images_fps)

    def visualize(self, i): 
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image     


def check_cityscapes_labels_found(images, labels): 
    """
    For CityScapes it is rather easy, as each image has a corresponding label, no errors will be found.
    If some file is missing it will print out which sample is missing. 
    
    """
    for img, lbl in zip(images, labels): 
        im = img.split('images/')[-1]
        lb = lbl.split('labels/')[-1]
        
        if not os.path.isfile(img): 
            Warning('Image file not found: {}'.format(img))
        if not os.path.isfile(lbl): 
            Warning('Label file not found: {}'.format(lbl))

        if not im.replace('_leftImg8bit.png', '') == lb.replace('_gtFine_labelIds.png', ''):
            print('Missmatch here\n {}\n {}: '.format(im, lb))
            return -1 
    return 0


def kitti_create_list_of_samples(train_txt=None, val_txt=None, base_path='/', subset=100):
    if train_txt is None or val_txt is None:
        raise ValueError('No label files sent')
    
    train_images, train_labels, validation_images, validation_labels = [], [], [], []
    
    with open(train_txt) as f:
        lines = f.readlines()
        
        for line in lines: 
            image, label = line.strip().split(' ')
            train_images.append(os.path.join(base_path, image))
            train_labels.append(os.path.join(base_path, label))
        
    with open(val_txt) as f: 
        lines = f.readlines()
        for line in lines: 
            image, label = line.strip().split(' ')
            validation_images.append(os.path.join(base_path, image))
            validation_labels.append(os.path.join(base_path, label))
    
    subset_divider = int(100/subset)
    return train_images[::subset_divider], train_labels[::subset_divider], validation_images[::subset_divider], validation_labels[::subset_divider]


def check_kitti360_labels_found(images, labels): 
    """
    Go through all the images from the valdation and train text files. Make sure they are found 
    in the Data storage folder. Also, make sure all images have a corresponding label file. 
    
    """
    for img, lbl in zip(images, labels): 
        im = img.split('data_2d_raw/')[-1]
        lb = lbl.split('data_2d_semantics/train/')[-1]
        
        im = im.replace('data_rect/', '')
        lb = lb.replace('semantic/', '')
        if not os.path.isfile(img): 
            Warning('Image file not found: {}'.format(img))
        if not os.path.isfile(lbl): 
            Warning('Label file not found: {}'.format(lbl))
        
        if not im == lb:
            print('Missmatch here\n {}\n {}: '.format(img, lbl))
            return -1 
    return 0


def cityscapes_loader(preprocessing_fn, augmentation_fn, dataset='validation', num_workers=4): 
    from utils.parameters import (
        CITYSCAPES_TRAINING_IMAGES, CITYSCAPES_TRAINING_LABELS,
        CITYSCAPES_VALIDATION_IMAGES, CITYSCAPES_VALIDATION_LABELS, 
        CITYSCAPES_TEST_IMAGES, CITYSCAPES_TEST_LABELS, 
        CLASSES, DEVICE, CLASS_DISTRIBUTION_FOLDER, 
    )
    NUM_CLASSES = len(CLASSES)

    # Assign the correct paths to the dataset
    training_images = sorted(list_dir_recursive(CITYSCAPES_TRAINING_IMAGES))
    training_labels = sorted(list_dir_recursive(CITYSCAPES_TRAINING_LABELS, 'labelIds.png'))

    validation_images = sorted(list_dir_recursive(CITYSCAPES_VALIDATION_IMAGES))
    validation_labels = sorted(list_dir_recursive(CITYSCAPES_VALIDATION_LABELS, 'labelIds.png'))

    test_images = sorted(list_dir_recursive(CITYSCAPES_TEST_IMAGES))
    test_labels = sorted(list_dir_recursive(CITYSCAPES_TEST_LABELS, 'labelIds.png'))

    # Check that the image-label pairs are found 
    check_cityscapes_labels_found(training_images, training_labels)
    check_cityscapes_labels_found(validation_images, validation_labels)
    check_cityscapes_labels_found(test_images, test_labels)
    
    # Construct the loader
    if dataset == 'train': 
        images = training_images
        labels = training_labels
    elif dataset == 'validation': 
        images = validation_images
        labels = validation_labels
    elif 'test': 
        images = test_images
        labels = test_labels
    
    dataset = CityScapes(
        images, 
        labels, 
        augmentation=augmentation_fn, 
        preprocessing=preprocessing_fn, 
        classes=CLASSES,
        preload=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    return loader 


def kitti_loader(preprocessing_fn, augmentation_fn, dataset='validation', num_workers=4, subset=100, individual=False): 
    # KITTI and Cityscapes uses the same dataloader. 
    from utils.parameters import (
        KITTI_TRAIN_FILE, KITTI_VALIDATION_FILE, KITTI_DATA_DIR, CLASSES, DEVICE 
    )
    NUM_CLASSES = len(CLASSES)

    # Subset is used if you just want a portion of the dataset. It takes every subset/100'th sample (%)
    kitti_train_images, kitti_train_labels, kitti_validation_images, kitti_validation_labels = kitti_create_list_of_samples(
        KITTI_TRAIN_FILE, KITTI_VALIDATION_FILE, KITTI_DATA_DIR, subset=subset
    )  

    # Check that the image-label pairs are found 
    check_kitti360_labels_found(kitti_train_images, kitti_train_labels)
    check_kitti360_labels_found(kitti_validation_images, kitti_validation_labels)
    
    # Construct the loader
    if dataset == 'train': 
        images = kitti_train_images
        labels = kitti_train_labels
    elif dataset == 'validation': 
        images = kitti_validation_images
        labels = kitti_validation_labels
    
    dataset = CityScapes(
        images, 
        labels, 
        augmentation=augmentation_fn, 
        preprocessing=preprocessing_fn, 
        classes=CLASSES,
        preload=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    return loader 
