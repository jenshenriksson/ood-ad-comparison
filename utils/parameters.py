import sys, os 
sys.path.append('..')
sys.path.append(os.getcwd())

import utils.utils as CU
import utils.dataloaders as CD


# Cityscapes
CITYSCAPES_TRAINING_IMAGES = '/mnt/ml-data-storage/jens/CityScapes/images/train'
CITYSCAPES_TRAINING_LABELS = CITYSCAPES_TRAINING_IMAGES.replace('images', 'labels')
CITYSCAPES_VALIDATION_IMAGES = '/mnt/ml-data-storage/jens/CityScapes/images/val'
CITYSCAPES_VALIDATION_LABELS = CITYSCAPES_VALIDATION_IMAGES.replace('images', 'labels')
CITYSCAPES_TEST_IMAGES = '/mnt/ml-data-storage/jens/CityScapes/images/test'
CITYSCAPES_TEST_LABELS = CITYSCAPES_TEST_IMAGES.replace('images', 'labels')

# BDD100K locations 
BDD100K_DATA_DIR = '/mnt/ml-data-storage/jens/BDD100K/images'
BDD100K_LABEL_DIR = BDD100K_DATA_DIR.replace('images', 'labels')
BDD100K_USA_SET_IMAGES = os.path.join(BDD100K_DATA_DIR, 'test1')
BDD100K_USA_SET_LABELS = BDD100K_USA_SET_IMAGES.replace('images', 'labels')
BDD100K_ISRAEL_SET_IMAGES = os.path.join(BDD100K_DATA_DIR, 'test2')
BDD100K_ISRAEL_SET_LABELS = BDD100K_ISRAEL_SET_IMAGES.replace('images', 'labels')

# KITTI-360
KITTI_DATA_DIR = '/mnt/ml-data-storage/jens/KITTI-360/Data'
KITTI_TRAIN_FILE = os.path.join(KITTI_DATA_DIR, 'data_2d_semantics/train', '2013_05_28_drive_train_frames.txt')
KITTI_VALIDATION_FILE = os.path.join(KITTI_DATA_DIR, 'data_2d_semantics/train', '2013_05_28_drive_val_frames.txt')

# A2D2
A2D2_DATA_DIR = '/mnt/ml-data-storage/jens/A2D2/camera_lidar_semantic'




# Classes in both Cityscapes, BDD100K, KITTI-360 and (partly) in A2D2
CLASSES = ['road', 'sidewalk', 'building', 'wall', 
       'fence', 'pole', 'traffic light', 'traffic sign', 
       'vegetation', 'terrain', 'sky', 
       'person', 'rider', 
       'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']


# Preprocessing dimensions
PSPNET_IMAGE_DIMS = {
       'cityscapes-train': {'width': 2041, 'height': 1017},
       'cityscapes-val': {'width': 2041, 'height': 1017},
       'bdd-israel': {'width': 1273, 'height': 713},
       'bdd-usa': {'width': 1273, 'height': 713},
       'kitti-train': {'width': 1401, 'height': 369},
       'kitti-val': {'width': 1401, 'height': 369},
       'a2d2': {'width': 1913, 'height': 1201}
}

IMAGE_DIMS = {
       'cityscapes-train': {'width': 2048, 'height': 1024},
       'cityscapes-val': {'width': 2048, 'height': 1024},
       'bdd-israel': {'width': 1280, 'height': 720},
       'bdd-usa': {'width': 1280, 'height': 720},
       'kitti-train': {'width': 1408, 'height': 376},
       'kitti-val': {'width': 1408, 'height': 376},
       'a2d2': {'width': 1920, 'height': 1208}
}

# Extra
DEVICE = 'cuda'
CLASS_DISTRIBUTION_FOLDER = './results/class_distribution'
ACCURACY_LOCATION = './results/accuracy'
MD_FOLDER = './results/mahalanobis'
