import os 
import matplotlib.pyplot as plt 
from tqdm import tqdm 
import numpy as np 

def print_accuracy(intersection_per_class, total_union_per_class, classes, print_all=True):
    if print_all: 
        print('Class ' + '-'* 11 + ' IoU ' + '-' * 9 )
        for cor, tot, cls in zip(intersection_per_class, total_union_per_class, classes):
            print('{:.15s}: {:.2f}%\t\t'.format(cls+' '*15, 100.0*cor.cpu().numpy()/(tot.cpu().numpy())))
    print(
        '{:.15s}: {:.2f}%\t\t'.format(
            'TOTAL'+' '*15, 100.0*intersection_per_class.cpu().numpy().sum()/total_union_per_class.cpu().numpy().sum() 
        )
    )


def compute_pixel_distribution(dataset=None, verbose=True): 
    num_classes = len(dataset.class_values)
    distribution = [0] * num_classes
    
    for i in tqdm(range(len(dataset))): 
        _, lbl = dataset[i]
        pixels = lbl.shape[0] * lbl.shape[1]
        for j in range(num_classes): 
            distribution[j] += lbl[:, :, j].sum() / pixels
    
    for j in range(num_classes): 
        distribution[j] = distribution[j] / len(dataset)
    if verbose: 
        for cls, prc in zip(dataset.CLASSES, distribution):
            print("{}: {}%".format(cls, np.round(prc*100,2)))
        print("Sum: {}".format(100.0*np.sum(distribution)))
    return distribution


def list_dir_recursive(input_path, endswith=None):
    '''
    Find all files in a given folder, in a recursive manner. 
    Returns the list of all file paths. 
    
    Optional: add a filetype ending. e.g. "png" 
    '''
    image_files = []
    for root, dirs, files in os.walk(input_path, topdown=True):
        for f in files: 
            if not endswith or f.endswith(endswith):
                image_files.append(os.path.join(root, f))
    return image_files




def visualize(**images):
    """
    Plot image, ground truth and prediction in one row.
    Code from https://github.com/qubvel/segmentation_models.pytorch ! 
    """
    
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
    
    

    

