import cv2 
import numpy as np

import torch 
from torch.utils.data import Dataset as Dataset
from torch.utils.data import DataLoader

import os 

import sys
sys.path.append('..')
import time
from utils.utils import list_dir_recursive

from tqdm.notebook import tqdm
import albumentations as albu
from albumentations.augmentations.geometric.resize import Resize




class A2D2(Dataset):

    """
    Audi Autonomous Driving Dataset (A2D2). Read images, apply augmentation and preprocessing transformations.
    DataLoader modified from https://github.com/qubvel/segmentation_models.pytorch 
    
    Information regarding CLASSES available from class_list.json included in the dataset. 
    
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
        "unlabeled",
        "car",  
        "bicycle",  
        "person",  # Renamed from pedestrian > person 
        "truck",  # Contains busses, thus will give an error in experiments. 
        "motorcycle",  # Renamed from small vehicle > motorcycle
        "traffic light",  # Renamed from traffic signals > traffic light 
        "traffic sign",  
        "utility vehicle",  # Ignored
        "sidebars",  # Ignored
        "speed bumper",  # Ignored
        "curbstone",  
        "solid line",  # Merged with Road
        "irrelevant signs",  
        "road blocks",  
        "tractor", 
        "non-drivable street",  # Merged with Road
        "zebra crossing",  # Merged with Road
        "obstacles", 
        "pole",  # Renamed from poles > pole 
        "restricted area",  # Ignored
        "animals",  # Ignored
        "fence",  # Renamed from grid structure > fence 
        "signal corpus",  # Ignored
        "drivable cobblestone",  # Merged with Road
        "electronic traffic",  # Ignored
        "slow driving area",  # Ignored
        "vegetation",  # Renamed from nature object > vegetation 
        "parking",  # Renamed from parking area > parking 
        "sidewalk", 
        "ego vehicle", 
        "painted driving instruction",  # Merged with Road
        "traffic guide", 
        "dashed line",  # Merged with Road
        "road",  # Renamed from rd normal street > road 
        "sky",  
        "building", 
        "blurred area",  # Ignored
        "rain dirt",  # Ignored
        "wall",  # Included, as exists in BDD100K
        "terrain",  # Included, as exists in BDD100K
        "bus",  # Included, as exists in BDD100K
        "train",  # Included, as exists in BDD100K
        "rider",  # Included, as exists in BDD100K
    ]

    INSTANCE_LABELS = {
        (255, 0, 0): 1,
        (200, 0, 0): 1,
        (150, 0, 0): 1,
        (128, 0, 0): 1,
        (182, 89, 6): 2,
        (150, 50, 4): 2,
        (90, 30, 1): 2,
        (90, 30, 30): 2,
        (204, 153, 255): 3,
        (189, 73, 155): 3,
        (239, 89, 191): 3,
        (255, 128, 0): 4,
        (200, 128, 0): 4,
        (150, 128, 0): 4,
        (0, 255, 0): 5,
        (0, 200, 0): 5,
        (0, 150, 0): 5,
        (0, 128, 255): 6,
        (30, 28, 158): 6,
        (60, 28, 100): 6,
        (0, 255, 255): 7,
        (30, 220, 220): 7,
        (60, 157, 199): 7,
        (255, 255, 0): 8,
        (255, 255, 200): 8,
        (233, 100, 0): 9,
        (110, 110, 0): 10,
        (128, 128, 0): 11,
        (255, 193, 37): 34,     # Changed from 12 to 34
        (64, 0, 64): 13,        
        (185, 122, 87): 14,
        (0, 0, 100): 15,
        (139, 99, 108): 34,     # Changed from 16 to 34
        (210, 50, 115): 34,     # Changed from 17 to 34
        (255, 0, 128): 18,
        (255, 246, 143): 19,
        (150, 0, 150): 20,
        (204, 255, 153): 21,
        (238, 162, 173): 22,
        (33, 44, 177): 23,
        (180, 50, 180): 34,     # Changed from 24 to 34
        (255, 70, 185): 25,
        (238, 233, 191): 26,
        (147, 253, 194): 27,
        (150, 150, 200): 28,
        (180, 150, 200): 29,
        (72, 209, 204): 30,
        (200, 125, 210): 34,    # Changed from 31 to 34
        (159, 121, 238): 32,
        (128, 0, 255): 34,      # Changed from 33 to 34
        (255, 0, 255): 34,
        (135, 206, 255): 35,
        (241, 230, 255): 36,
        (96, 69, 143): 37,
        (53, 46, 82): 38
    }  

    def __init__(
            self, 
            images, 
            labels, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.images_fps = images
        self.masks_fps = labels
        
        # convert str names to class values on masks
        if not classes: 
            classes = self.CLASSES
        
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        
    def create_instance_labels(self, label_image): 
        mask = np.zeros((label_image.shape[0], label_image.shape[1]))

        for key in self.INSTANCE_LABELS.keys(): 
            red_img = np.logical_and(np.logical_and((label_image[:,:,0] == key[0]), (label_image[:,:,1] == key[1])), (label_image[:,:,2] == key[2])) 
            mask[red_img] = self.INSTANCE_LABELS[key]

        return mask
    
    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label_img = cv2.imread(self.masks_fps[i])
        label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB)
        mask = self.create_instance_labels(label_img) 

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

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

class A2D2_ORIGINAL(Dataset):

    """
    Audi Autonomous Driving Dataset (A2D2). Read images, apply augmentation and preprocessing transformations.
    DataLoader modified from https://github.com/qubvel/segmentation_models.pytorch 
    
    Information regarding CLASSES available from class_list.json included in the dataset. 
    
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
        "unlabeled",
        "car",
        "bicycle",
        "pedestrian",
        "truck",
        "small vehicle",
        "traffic signal",
        "traffic sign",
        "utility vehicle",
        "sidebars",
        "speed bumper",
        "curbstone",
        "solid line",
        "irrelevant signs",
        "road blocks",
        "tractor",
        "non-drivable street",
        "zebra crossing",
        "obstacles",
        "poles",
        "restricted area",
        "animals",
        "grid structure",
        "signal corpus",
        "drivable cobblestone",
        "electronic traffic",
        "slow driving area",
        "nature object",
        "parking area",
        "sidewalk",
        "ego vehicle",
        "painted driving instruction",
        "traffic guide",
        "dashed line",
        "rd normal street",
        "sky",
        "building",
        "blurred area",
        "rain dirt"
    ]
    
    INSTANCE_LABELS = {
        (255, 0, 0): 1,
        (200, 0, 0): 1,
        (150, 0, 0): 1,
        (128, 0, 0): 1,
        (182, 89, 6): 2,
        (150, 50, 4): 2,
        (90, 30, 1): 2,
        (90, 30, 30): 2,
        (204, 153, 255): 3,
        (189, 73, 155): 3,
        (239, 89, 191): 3,
        (255, 128, 0): 4,
        (200, 128, 0): 4,
        (150, 128, 0): 4,
        (0, 255, 0): 5,
        (0, 200, 0): 5,
        (0, 150, 0): 5,
        (0, 128, 255): 6,
        (30, 28, 158): 6,
        (60, 28, 100): 6,
        (0, 255, 255): 7,
        (30, 220, 220): 7,
        (60, 157, 199): 7,
        (255, 255, 0): 8,
        (255, 255, 200): 8,
        (233, 100, 0): 9,
        (110, 110, 0): 10,
        (128, 128, 0): 11,
        (255, 193, 37): 12,
        (64, 0, 64): 13,
        (185, 122, 87): 14,
        (0, 0, 100): 15,
        (139, 99, 108): 16,
        (210, 50, 115): 17,
        (255, 0, 128): 18,
        (255, 246, 143): 19,
        (150, 0, 150): 20,
        (204, 255, 153): 21,
        (238, 162, 173): 22,
        (33, 44, 177): 23,
        (180, 50, 180): 24,
        (255, 70, 185): 25,
        (238, 233, 191): 26,
        (147, 253, 194): 27,
        (150, 150, 200): 28,
        (180, 150, 200): 29,
        (72, 209, 204): 30,
        (200, 125, 210): 31,
        (159, 121, 238): 32,
        (128, 0, 255): 33,
        (255, 0, 255): 34,
        (135, 206, 255): 35,
        (241, 230, 255): 36,
        (96, 69, 143): 37,
        (53, 46, 82): 38
    }  
    
    
    def __init__(
            self, 
            images, 
            labels, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.images_fps = images
        self.masks_fps = labels
        
        # convert str names to class values on masks
        if not classes: 
            classes = self.CLASSES
        
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        
    def create_instance_labels(self, label_image): 
        mask = np.zeros((label_image.shape[0], label_image.shape[1]))

        for key in self.INSTANCE_LABELS.keys(): 
            red_img = np.logical_and(np.logical_and((label_image[:,:,0] == key[0]), (label_image[:,:,1] == key[1])), (label_image[:,:,2] == key[2])) 
            mask[red_img] = self.INSTANCE_LABELS[key]

        return mask
    
    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label_img = cv2.imread(self.masks_fps[i])
        label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB)
        mask = self.create_instance_labels(label_img) 

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

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

def check_a2d2_labels_found(images, labels): 
    """
    Go through all the images from the valdation and train text files. Make sure they are found 
    in the Data storage folder. Also, make sure all images have a corresponding label file. 
    
    """
    for img, lbl in zip(images, labels): 
        im = img.split('/')[-1]
        lb = lbl.split('/')[-1]
        
        im = im.replace('camera', '')
        lb = lb.replace('label', '') 
        if not os.path.isfile(img): 
            Warning('Image file not found: {}'.format(img))
        if not os.path.isfile(lbl): 
            Warning('Label file not found: {}'.format(lbl))
        
        if not im == lb:
            print('Missmatch here\n {}\n {}: '.format(img, lbl))
            return -1
    return 0 


def a2d2_create_list_of_samples(a2d2_data_dir, subset=100): 
    a2d2_images = []
    a2d2_labels = []
    all_a2d2_files = list_dir_recursive(a2d2_data_dir, '.png')
    for a in all_a2d2_files: 
        if '_label_' in a: 
            a2d2_labels.append(a)
        else: 
            a2d2_images.append(a)
    
    a2d2_images = sorted(a2d2_images)
    a2d2_labels = sorted(a2d2_labels)
    
    # Reduce data if one so desires 
    subset_divider = int(100/subset)
    return a2d2_images[::subset_divider], a2d2_labels[::subset_divider]


def a2d2_loader(preprocessing_fn, augmentation_fn, num_workers=4, subset=100, individual=False): 
    from utils.parameters import (
        A2D2_DATA_DIR, CLASSES, DEVICE 
    )
    NUM_CLASSES = len(CLASSES)

    # Subset is used if you just want a portion of the dataset. It takes every subset/100'th sample (%)
    a2d2_images, a2d2_labels = a2d2_create_list_of_samples(A2D2_DATA_DIR, subset=subset)

    # Check that the image-label pairs are found 
    check_a2d2_labels_found(a2d2_images, a2d2_labels)

    # Construct the loader
    images = a2d2_images
    labels = a2d2_labels
    
    dataset = A2D2(
        images, 
        labels, 
        augmentation=augmentation_fn, 
        preprocessing=preprocessing_fn, 
        classes=CLASSES,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    return loader 