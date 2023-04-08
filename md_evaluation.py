import argparse
import os
import gc 
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
from utils.metrics import *

import models.deeplabv3
import models.load_models as load_models 

from utils.preprocessing import (
    get_preprocessing_fn, get_validation_augmentation, 
    get_preprocessing, pspnet_augmentation, 
)
from utils.parameters import (
    CLASSES, DEVICE, PSPNET_IMAGE_DIMS, IMAGE_DIMS, 
    CLASS_DISTRIBUTION_FOLDER, ACCURACY_LOCATION, MD_FOLDER

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
            
        # Load in the class-conditional Gaussian distribution.  
        class_distribution_filename = os.path.join(CLASS_DISTRIBUTION_FOLDER, '{}_class_distribution.pickle'.format(model_name))
        mean_vectors, inverse_cov_vectors, cov_vectors = mean_and_inv_variances(class_distribution_filename, NUM_CLASSES)

        # Evaluate each model on the desired dataset. Default is only on Cityscapes 
        for dataset_name in args.datasets: 
            mahalanobis_results_folder =  os.path.join('{}{}'.format(MD_FOLDER, args.thresholds), model_name, dataset_name)
            mahalanobis_results = os.path.join(mahalanobis_results_folder, 'mahalanobis_results.npy')
            if os.path.isfile(mahalanobis_results) and not args.again: 
                print("Already completed {}-{}. If you want to run again, add --again".format(model_name, dataset_name))
                continue
            os.makedirs(mahalanobis_results_folder, exist_ok=True)

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
            results = extract_masks_from_dataset(dataset_loader, model, mean_vectors, inverse_cov_vectors, device=DEVICE, mode=args.mode, threshold_points=args.thresholds)

            # Save the results 
            np.save(mahalanobis_results, results)

            # results is a M x N x 6 vector, where M is number of images. We're using N thresholding points
            # ranging between th.min() to th.max(). The 6 variables per run is tp, fp, fn, tn, risk, coverage
            risk=results[:, :, 4]  
            coverage=results[:, :, 5]
            x = np.linspace(0, 1, args.thresholds)

            plt.plot(x, risk.mean(axis=0), '.', color='red')
            plt.xlabel('Threshold')
            plt.ylabel('Risk (1-IoU)')
            plt.savefig(os.path.join(mahalanobis_results_folder, 'risk_vs_threshold.pdf'))

            plt.figure()
            plt.plot(x, coverage.mean(axis=0), '.', color='red')
            plt.xlabel('Threshold')
            plt.ylabel('Coverage (%)')
            plt.savefig(os.path.join(mahalanobis_results_folder, 'coverage_vs_threshold.pdf'))

            minimum_risk = risk.mean(axis=0).argmin()
            print(np.argmin(risk.mean(axis=0)))

            plt.figure()
            plt.plot(coverage.mean(axis=0), risk.mean(axis=0), '.', color='red', label='Risk-coverage')
            plt.xlabel('Coverage (%)')
            plt.ylabel('Risk (1-IoU)')
            plt.legend()
            plt.savefig(os.path.join(mahalanobis_results_folder, 'risk_vs_coverage.pdf'))

        del model
        gc.collect()
        torch.cuda.empty_cache()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example script with optional arguments')
    parser.add_argument('--models', type=str, nargs='+', default=['pspnet'], help='List of models (mobilenet, pspnet, resnet101)')
    parser.add_argument('--datasets', type=str, nargs='+', default=['cityscapes-val'], help='list of datasets (a2d2, bdd-israel bdd-usa, cityscapes-train cityscapes-val, kitti-train kitti-val)')
    parser.add_argument('--verbose', action='store_true', default=False, help='print verbose output')
    parser.add_argument('--mode', type=str, default=None, help='Sets the output layer to softmax, sigmoid or nothing (model).')
    parser.add_argument('--again', action='store_true', default=False, help='Run accuracy scores again. Otherwise skips those that are done')
    parser.add_argument('--thresholds', type=int, default=100, help='Optional to set the amount of threshold variations for the AUROC-scores')
    parser.add_argument('--subset', type=int, default=100, help='Optional to take a percentage subset for A2D2 and KITTI')
    parser.add_argument('--individual', action='store_true', default=False, help='Optional to run A2D2/KITTI-runs individual')
    
    args = parser.parse_args()

    # Printing a summary of parameters selected
    if args.models:
        print(f'Model file: {args.models}')

    if args.datasets:
        print(f'Dataset files: {args.datasets}')

    if args.thresholds: 
        print(f'Dataset files: {args.thresholds}')
    
    main(args)