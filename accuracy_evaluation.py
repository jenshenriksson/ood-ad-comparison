import argparse
import os
import time 
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import matplotlib.pyplot as plt 
import pickle 
from tqdm.notebook import tqdm

import albumentations as albu
from albumentations.augmentations.geometric.resize import Resize

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

import utils.utils as CU
from utils.dataloaders.cityscapes import cityscapes_loader, kitti_loader 
from utils.dataloaders.bdd100k import bdd100k_loader 
from utils.dataloaders.audi import a2d2_loader 
from utils.metrics import iou_test_per_class, correct_classified_pixel_distribution 

import models.deeplabv3
import models.load_models as load_models 

from utils.preprocessing import (
    get_preprocessing_fn, get_validation_augmentation, 
    get_preprocessing, pspnet_augmentation, 
)
from utils.parameters import (
    CLASSES, DEVICE, CLASS_DISTRIBUTION_FOLDER, 
    PSPNET_IMAGE_DIMS, IMAGE_DIMS, ACCURACY_LOCATION, 

)
NUM_CLASSES = len(CLASSES)


def main(args):
    # Construct a normalization function call. Based on ImageNet normalization params. 
    preprocessing_fn = get_preprocessing_fn()  
    
    
    # Load in the specific models. Currently covering DeepLabV3 (MobileNet and ResNet101) and PSPNet
    for model_name in args.models: 
        if model_name=='pspnet': 
            model = load_models.pspnet(NUM_CLASSES)
        elif model_name == 'mobilenet': 
            model = load_models.deeplab_mobilenet(NUM_CLASSES)
        elif model_name == 'resnet101': 
            model = load_models.deeplab_resnet101(NUM_CLASSES)
        
        # Prepare model for evaluation. 
        model.to(DEVICE)
        model.eval() 

        # Evaluate each model on the desired dataset. Default is only on Cityscapes 
        for dataset_name in args.datasets: 
            if model_name == 'pspnet':
                width = PSPNET_IMAGE_DIMS[dataset_name]['width']
                height = PSPNET_IMAGE_DIMS[dataset_name]['height']
            else: 
                width = IMAGE_DIMS[dataset_name]['width']
                height = IMAGE_DIMS[dataset_name]['height']

            if dataset_name == 'cityscapes-train': 
                dataset_loader = cityscapes_loader(preprocessing_fn=get_preprocessing(preprocessing_fn), augmentation_fn=get_validation_augmentation(height=height, width=width), dataset='train')
            elif dataset_name == 'cityscapes-val': 
                dataset_loader = cityscapes_loader(preprocessing_fn=get_preprocessing(preprocessing_fn), augmentation_fn=get_validation_augmentation(height=height, width=width), dataset='validation')
            elif dataset_name == 'bdd-usa':  # The BDD100K subset of data from USA.
                dataset_loader = bdd100k_loader(preprocessing_fn=get_preprocessing(preprocessing_fn), augmentation_fn=get_validation_augmentation(height=height, width=width), dataset='usa')
            elif dataset_name == 'bdd-israel':  # The BDD100K Subset of data from Israel 
                dataset_loader = bdd100k_loader(preprocessing_fn=get_preprocessing(preprocessing_fn), augmentation_fn=get_validation_augmentation(height=height, width=width), dataset='israel')
            elif dataset_name == 'kitti-train':
                dataset_loader = kitti_loader(preprocessing_fn=get_preprocessing(preprocessing_fn), augmentation_fn=get_validation_augmentation(height=height, width=width), dataset='train', subset=args.subset)
            elif dataset_name == 'kitti-val':
                dataset_loader = kitti_loader(preprocessing_fn=get_preprocessing(preprocessing_fn), augmentation_fn=get_validation_augmentation(height=height, width=width), dataset='validation', subset=args.subset, individual=args.individual)
            elif dataset_name == 'a2d2':
                dataset_loader = a2d2_loader(preprocessing_fn=get_preprocessing(preprocessing_fn), augmentation_fn=get_validation_augmentation(height=height, width=width), subset=args.subset, individual=args.individual)
            else:
                Warning('Unknown dataset: {}'.format(dataset_name))
                continue
                
            # Do the actual evaluation 
            if args.mode == 'accuracy': 
                if os.path.isfile(os.path.join(ACCURACY_LOCATION, '{}_{}_meanIoU.txt'.format(model_name, dataset_name))) and not args.again: 
                    print("Already completed {}-{}. If you want to run again, add --again".format(model_name, dataset_name))
                    continue
                
                start_time = time.time()
                intersection_per_class, total_union_per_class = iou_test_per_class(
                    dataset_loader, model, num_classes=len(CLASSES), device=DEVICE, mode='softmax'
                )
                elapsed_time = (time.time() - start_time) / 60  # Elapsed time in minutes. 

                # Always save results to file. If verbose, also print it out. 
                os.makedirs(ACCURACY_LOCATION, exist_ok=True)
                with open(os.path.join(ACCURACY_LOCATION, '{}_{}_meanIoU.txt'.format(model_name, dataset_name)), 'w') as f: 
                    print(f'Run took: {elapsed_time:.2f} minutes', file=f)
                    print('Class ' + '-'* 11 + ' IoU ' + '-' * 2 , file=f)
                    for cor, tot, cls in zip(intersection_per_class, total_union_per_class, CLASSES):
                        print('{:.15s}: {:.2f}%\t\t'.format(cls+' '*15, 100.0*cor.cpu().numpy()/(tot.cpu().numpy())), file=f)
                    print('{:.15s}: {:.2f}%\t\t'.format('TOTAL'+' '*15, 100.0*intersection_per_class.cpu().numpy().sum()/total_union_per_class.cpu().numpy().sum()) , file=f)
                
                if args.verbose:   
                    print('Results of {} on {}'.format(model_name, dataset_name)) 
                    print('{:.15s}: {:.2f}%\t\t'.format('TOTAL'+' '*15, 100.0*intersection_per_class.cpu().numpy().sum()/total_union_per_class.cpu().numpy().sum()))

            elif args.mode == 'means': 
                os.makedirs(CLASS_DISTRIBUTION_FOLDER, exist_ok=True)
                if dataset_name != 'cityscapes-train': 
                    print("Can only compute means on Cityscapes training set")
                    exit(0)

                # Is it already done? or should be done again 
                distribution_file_name = os.path.join(CLASS_DISTRIBUTION_FOLDER, '{}_class_distribution.pickle'.format(model_name))
                if os.path.isfile(distribution_file_name) and not args.again:
                    print(distribution_file_name)
                    print("Already completed {}. If you want to run again, add --again".format(model_name))
                    continue

                # Compute the class-conditional Gaussian Distribution 
                with open(distribution_file_name, 'wb') as file: 
                    correct_pixels = correct_classified_pixel_distribution(model, dataset_loader, device=DEVICE, num_classes=len(CLASSES))    
                    pickle.dump(correct_pixels, file, protocol=pickle.HIGHEST_PROTOCOL)

    # Printing a summary of parameters selected
    if args.models:
        print(f'Model file: {args.models}')

    if args.datasets:
        print(f'Dataset files: {args.datasets}')
    
    if args.mode:
        print(f'Dataset files: {args.mode}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example script with optional arguments')
    parser.add_argument('--models', type=str, nargs='+', default=['pspnet'], help='List of models (mobilenet, pspnet, resnet101)')
    parser.add_argument('--mode', type=str, default='accuracy', help='What to do? Computes accuracy, or distribution')
    parser.add_argument('--datasets', type=str, nargs='+', default=['cityscapes-val'], help='list of datasets (a2d2, bdd-israel bdd-usa, cityscapes-train cityscapes-val, kitti-train kitti-val)')
    parser.add_argument('--verbose', action='store_true', default=False, help='print verbose output')
    parser.add_argument('--again', action='store_true', default=False, help='Run accuracy scores again. Otherwise skips those that are done')
    parser.add_argument('--subset', type=int, default=100, help='Optional to take a percentage subset for A2D2 and KITTI')
    parser.add_argument('--individual', action='store_true', default=False, help='Optional to run A2D2/KITTI-runs individual')
    
    args = parser.parse_args()
    main(args)