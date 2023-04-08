import cv2 
import numpy as np
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


class BDD100K(Dataset):
    """    
    DataLoader for the Semantic Segmentation subset of BDD100K. 
    Read images, apply augmentation and preprocessing transformations.
    DataLoader modified from https://github.com/qubvel/segmentation_models.pytorch 
    
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
        'road', 
        'sidewalk', 
        'building', 
        'wall', 
        'fence',  
        'pole', 
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
        'train', 
        'motorcycle', 
        'bicycle',  
        'unknown', 
    ] 
    
    def __init__(
            self, 
            images, 
            labels, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
            preload = False, 
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
        if not self.preload: 
            # read data
            image = cv2.imread(self.images_fps[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.masks_fps[i], 0)
        else: 
            image = self.preloadedImages[i]
            mask = self.preloadedMasks[i]

        # extract certain classes from mask + convert to one-hot. 
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
    


def check_bdd100k_images_found(images, labels): 
    """
    For CityScapes it is rather easy, as each image has a corresponding label, no errors will be found.
    If some file is missing it will print out which sample is missing. 
    
    """
    for img, lbl in zip(images, labels): 
        im = img.split('images/')[-1]
        lb = lbl.split('labels/')[-1]
        
        im = im.replace('.jpg', '')
        lb = lb.replace('-mask.png', '') 
        
        if not os.path.isfile(img): 
            Warning('Image file not found: {}'.format(img))
        if not os.path.isfile(lbl): 
            Warning('Label file not found: {}'.format(lbl))
        
        if not im == lb:
            print('Missmatch here\n {}\n {}: '.format(im, lb))
            return -1 
    return 0 


def bdd100k_loader(preprocessing_fn, augmentation_fn, dataset='usa', num_workers=4): 
    from utils.parameters import (
        BDD100K_USA_SET_IMAGES, BDD100K_USA_SET_LABELS,
        BDD100K_ISRAEL_SET_IMAGES, BDD100K_ISRAEL_SET_LABELS, 
        CLASSES, DEVICE
    )   
    NUM_CLASSES = len(CLASSES)

    # Assign the correct paths to the dataset
    bdd100k_usa_images = sorted(list_dir_recursive(BDD100K_USA_SET_IMAGES))
    bdd100k_usa_labels = sorted(list_dir_recursive(BDD100K_USA_SET_LABELS, '-mask.png'))

    bdd100k_israel_images = sorted(list_dir_recursive(BDD100K_ISRAEL_SET_IMAGES))
    bdd100k_israel_labels = sorted(list_dir_recursive(BDD100K_ISRAEL_SET_LABELS, '-mask.png'))

    # Check that the image-label pairs are found 
    check_bdd100k_images_found(bdd100k_usa_images, bdd100k_usa_labels)
    check_bdd100k_images_found(bdd100k_israel_images, bdd100k_israel_labels)
    
    # Construct the loader
    if dataset == 'usa': 
        images = bdd100k_usa_images
        labels = bdd100k_usa_labels
    elif dataset == 'israel': 
        images = bdd100k_israel_images
        labels = bdd100k_israel_labels
    
    dataset = BDD100K(
        images, 
        labels, 
        augmentation=augmentation_fn, 
        preprocessing=preprocessing_fn, 
        classes=CLASSES,
        preload=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    return loader 

