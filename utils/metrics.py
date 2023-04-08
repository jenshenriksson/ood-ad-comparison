import os 
import torch
import pickle 
import numpy as np 
from torch.utils.data import DataLoader
from tqdm import tqdm
import time 

def compute_iou(prediction, gt_mask, eps=1e-7, th_mask=None):  
    '''
    pd_mask: Prediction mask 
    gt_mask: Ground truth mask
    th_mask: Threshold mask 
    eps: epsilon to avoid zero division
    '''
    if th_mask is not None: 
        pr_mask = prediction > th_mask
    else: 
        pr_mask = (prediction.round())
    
    area_of_overlap = np.logical_and(gt_mask, pr_mask).sum()
    area_of_union = np.logical_or(gt_mask, pr_mask).sum() + eps 
    intersection_over_union = area_of_overlap / area_of_union

    return intersection_over_union


def compute_class_pixel_distribution(dataset, device=None, dims=(0, 1, 2)): 
    '''
    The compute_class_pixel_distribution function computes the pixel-wise distribution of classes in a given dataset. 
    
    It takes three arguments:
    dataset: a torch.utils.data.Dataset object containing the dataset to compute the distribution for.
    device: (optional) the device to perform computations on. If not provided, the computations will be performed on the CPU.
    dims: (optional) a tuple of dimensions to sum over when computing the distribution. 
    '''
    num_classes = len(dataset.class_values)
    samples_per_class = torch.zeros(num_classes).to(device)
    total_samples = torch.zeros(1).to(device)
    total_annotated = torch.zeros(1).to(device)

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
    itr = tqdm(loader, total=len(loader))
    
    with torch.no_grad(): 
        for _, lbl in itr: 
            lbl = lbl.to(device)
            this_frame = lbl.sum(dim=dims) 
            samples_per_class += this_frame / (lbl.numel() / num_classes) 
            total_annotated += lbl.sum() / (lbl.numel() / num_classes)
            total_samples += lbl.size(0)
  
    return samples_per_class, total_samples, total_annotated


def iou_test_per_class(loader, model, num_classes, device, mode=None, dims=(0, 2, 3), eps=1):
    '''The iou_test_per_class function calculates the intersection over union (IOU) metric for each 
    class in a dataset using a given model.

    It takes three arguments:
    loader: A data loader object that yields batches of images and labels
    model: A model object that takes images as input and outputs predictions
    num_classes: An integer representing the number of classes in the dataset
    device: The device (e.g. "cpu" or "cuda") on which the tensors should be stored
    mode: A string indicating which output adjustment is needed (Accepts: None, sigmoid and softmax)
    dims: A tuple of dimensions along which to sum the logical AND and logical OR operations (default: (0, 2, 3))
    eps: A small constant to add to the denominator of the IOU calculation to avoid division by zero (default: 1)
    
    The iou_test_per_class function returns a tuple containing the following: 
    intersection_per_class: A tensor of shape (num_classes,) representing the sum of intersections per class
    total_union_per_class: A tensor of shape (num_classes,) representing the sum of unions per class
    '''
    model.eval()
    intersection_per_class = torch.zeros(num_classes).to(device)
    total_union_per_class = torch.zeros(num_classes).to(device)
    ctr = 0 
    with torch.no_grad():
        itr = tqdm(loader, total=len(loader))
        for images, labels in itr: 
            images = images.to(device)
            labels = labels.to(device)

            if mode is not None: 
                if mode == 'sigmoid': 
                    preds = torch.sigmoid(model(images))
                elif mode == 'softmax': 
                    preds = model(images).softmax(dim=1)
                else: 
                    return "Wrong mode" 
            else: 
                preds = model(images)
                
            pd_mask = (preds > 0.5).float() 
            intersection_per_class += (torch.logical_and(pd_mask, labels)).sum(dim=dims)  # Reduces away dims, so only dim1 remains
            total_union_per_class += (torch.logical_or(pd_mask, labels)).sum(dim=dims)
            itr.set_postfix({'avg iou': (intersection_per_class.sum()/(total_union_per_class.sum()+eps)).cpu().numpy()})    
        
    return intersection_per_class, total_union_per_class


def class_existence(dataset, device=None, dims=(0, 1, 2)): 
    '''
    The class_existence function calculates the number of pixels belonging to each class for each image in a given dataset. 
    It does this by creating a tensor of shape (num_images, num_classes) that is initialized to zeros, 
    and then iterating over the dataset in a data loader, summing the number of pixels belonging 
    to each class for each image, and storing the result in the tensor. 
    The function takes the following inputs:

    dataset: A dataset object containing images and labels
    device: The device (e.g. "cpu" or "cuda") on which the tensors should be stored (default: None)
    dims: A tuple of dimensions along which to sum the pixels belonging to each class (default: (0, 1, 2))
    
    The class_existence function returns a tensor of shape (num_images, num_classes) 
    representing the number of pixels belonging to each class for each image in the dataset.
    '''
    num_classes = len(dataset.class_values)
    num_images = len(dataset)
    
    samples_per_image = torch.zeros(num_images, num_classes).to(device)

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
    itr = tqdm(loader, total=len(loader))
    
    with torch.no_grad(): 
        for idx, (_, lbl) in enumerate(itr): 
            lbl = lbl.to(device)
            samples_per_image[idx] += lbl.sum(dim=dims)
    return samples_per_image


def correct_classified_pixel_distribution(model, loader, device, num_classes, max_samples=1e6): 
    """
    Extracts the probability distribution of correctly classified pixels for each class in the dataset.
    Since some models are pre-set with an output layer (e.g., Softmax layer), and some don't, 
    the mode option allows to specify softmax and sigmoid, if none is selected it is assumed that the output layer
    is already in the model itself.  
    """

    sm = torch.nn.Softmax(dim=1)
    model.eval()
    model.to(device)

    # Instantiate an empty np array for each class in the dataset
    correct_pixels = [np.empty((num_classes,0))]*num_classes 

    with torch.no_grad(): 
        for images, labels in tqdm(loader, total=len(loader)):
            

            images = images.to(device)
            gt_mask = labels.squeeze() 
            
            # Apply the specified activation function to the model's output, if provided
            model_output = model(images).squeeze() 

            preds = model_output - model_output.min(1, keepdim=True)[0]
            preds /= preds.max(1, keepdim=True)[0]
            pd_mask = (preds > 0.5).float().cpu().numpy()
            
            num_completed = 0 
            # For each class in the dataset, compute the true positives and add them to the corresponding array
            for cls in range(num_classes): 
                if correct_pixels[cls].shape[1] > max_samples:  # Get 1 million pixels of each class (or more)
                    num_completed += 1 
                    continue

                # Extract probability vectors for every pixel that is correctly classified. 
                true_positives = np.logical_and(pd_mask[cls, :, :], gt_mask[cls, :, :])
                samples_found = true_positives.sum()
                if samples_found >= 1: 
                    while true_positives.sum() > 10000: 
                        hei, wid = true_positives.shape 
                        true_positives = np.logical_and(true_positives, torch.randint(0, 2, (hei, wid)))
                    correct_pixels[cls] = np.hstack([correct_pixels[cls], model_output[:,true_positives].cpu().numpy()])

            if num_completed == num_classes:
                break  # If we skipped all classes, we're done! 
    return correct_pixels


def mahalanobis_distance(sample, pr_mask, mean_vectors, inv_cov_vectors, max_dist=1.0): 
    '''
    gets a torch-prediction vector with size Height x width x channels, 
    and the mean and inverted covariance matrices for the model (pre-calculated)

    Returns a distance map, a.k.a the Thresholding matrix of size H*W. 

    '''
    height, width, channels = sample.shape
    threshold_matrix = np.zeros((height, width))
    
    # pr_softmax = sm(sample)
    # pr_mask = (pr_softmax.squeeze().numpy().round())

    for c in range(channels): 
        
        index_mask = pr_mask[:,:,c]==1
        left_side = np.dot(sample[index_mask] - mean_vectors[c, :], inv_cov_vectors[c,:,:])
        right_side = (sample[index_mask] - mean_vectors[c, :]).T
        distances = np.sqrt(np.einsum('ij, ji ->i', left_side, right_side))
        threshold_matrix[index_mask] = distances
    
    if max_dist is not None: 
        threshold_matrix[threshold_matrix > max_dist] = max_dist 
    
    return threshold_matrix


def extract_masks_from_dataset(test_dataset, model, mean_vectors, inv_covs, mode='softmax', device='cuda', threshold_points=50):

    sm = torch.nn.Softmax()
    ctr = 0 
    start_time = time.time() 

    results = np.zeros((len(test_dataset), threshold_points, 6))

    for i, (image_tensor, label_tensor) in tqdm(enumerate(test_dataset), total=len(test_dataset)): 
        image_tensor = image_tensor.to(device)

        with torch.no_grad(): 

            if mode is not None: 
                if mode == 'sigmoid': 
                    prediction = torch.sigmoid(model(image_tensor))
                elif mode == 'softmax': 
                    prediction = model(image_tensor).softmax(dim=1)
                else: 
                    return "Wrong mode" 
            else: 
                prediction = model(image_tensor)

            prediction = prediction.detach().squeeze()
            if torch.all(prediction <= 1.0): 
                pr_mask = prediction.round()  # Checks if something is > 0.5, i.e. a prediction.  
            else: 
                pr_mask = prediction.softmax(dim=0).round()

            pr_mask = pr_mask.cpu().numpy().transpose(1, 2, 0)
            prediction = prediction.cpu().numpy().transpose(1, 2, 0)

            # Convert for MD-metric 
            threshold_matrix = mahalanobis_distance(prediction, pr_mask, mean_vectors, inv_covs, max_dist=None)
            
            # Construct 3 Channel format: Threshold matrix; Argmax Prediction; Argmax Label
            label_tensor_transposed = label_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
            correct_flatten = np.sign((np.logical_and(pr_mask, label_tensor_transposed)).sum(axis=2)).astype('uint8')
            predicted_flatten = pr_mask.sum(axis=2)
            label_flatten = np.sign(label_tensor_transposed.sum(axis=2)).astype('uint8')
            binary_masks = np.array([correct_flatten, predicted_flatten, label_flatten]).astype('uint8')
            
            # Vary the threshold and get tp, fp etc. 
            results[i] = threshold_variation(binary_masks, threshold_matrix, threshold_points)

    print("Run took {} minutes".format((time.time()-start_time)/60))
    return results


def mean_and_inv_variances(class_distribution_filename, num_classes): 
    if os.path.isfile(class_distribution_filename):  
        correct_pixels = pickle.load(open(class_distribution_filename, 'rb'))

        mean_vectors = []
        covariance_vectors = []
        inverse_covariance_vectors = [] 

        for cls in range(num_classes): 
            mean_vectors.append(correct_pixels[cls].mean(axis=1))
            covariance = np.cov(correct_pixels[cls])
            covariance_vectors.append(covariance)
            inverse_covariance_vectors.append(np.linalg.inv(covariance))

        return np.array(mean_vectors), np.array(inverse_covariance_vectors), np.array(covariance_vectors)
    else: 
        print("Cannot find Class Conditional Gaussian Distribution files. Please check\n{}".format(class_distribution_filename))
        exit(0) 


def threshold_variation(binary_masks, threshold_matrix, threshold_points=50): 
    min_linspace = threshold_matrix.min()
    max_linspace = threshold_matrix.max()        
    result_vector = []
    for th in np.linspace(min_linspace, max_linspace, threshold_points): 
        tp, fp, fn, tn, risk, coverage = evaluation(binary_masks[0], binary_masks[1], binary_masks[2], threshold_matrix, th)
        result_vector.append([tp, fp, fn, tn, risk, coverage])

    return np.array(result_vector)

# def threshold_variation(binary_masks, th_matrices): 
#     first = True 
#     res = []
#     samples = len(binary_masks)

#     for bm, thm in tqdm(zip(binary_masks, th_matrices)): 
#         min_linspace = thm.min()
#         max_linspace = thm.max()        
#         tmp = []
#         for th in np.linspace(min_linspace, max_linspace): 
#             tp, fp, fn, tn, risk, coverage = evaluation(bm[0], bm[1], bm[2], thm, th)
#             # CU.visualize(tp = tp, fp = fp, fn = fn, tn = tn)
#             tmp.append([tp, fp, fn, tn, risk, coverage])

#         tmp = np.expand_dims(np.array(tmp), axis=0)
#         if first: 
#             first = False
#             res = tmp
#         else: 
#             res = np.vstack([res, tmp])


#     res = np.array(res)
#     return res


def evaluation(correct_flatten, predicted_flatten, original_label_flatten, threshold_matrix, th_value=1): 
    # Construct the inclusion mas "C" wherever we've had a prediction (if no prediction, Th is 0)
    include_mask = (threshold_matrix <= th_value) & (threshold_matrix != 0)

    # Construt Subset of True positives within include mask (A & B & C)
    correct_flatten = np.logical_and(correct_flatten, include_mask)

    # Get all predictions within the include mask
    predicted_flatten = (predicted_flatten > -1)
    predicted_flatten = np.logical_and(predicted_flatten, include_mask)
    label_flatten = np.logical_and(original_label_flatten, include_mask)
    
    tp = correct_flatten
    fp = np.logical_and(np.logical_xor(correct_flatten, predicted_flatten), predicted_flatten)
    fn = np.logical_and(np.logical_xor(correct_flatten, predicted_flatten), label_flatten)
    tn = (predicted_flatten == 0) & (label_flatten==0)
    
    IoU = tp.sum() / (np.logical_or(predicted_flatten, label_flatten).sum() + 1e-7)
    coverage = label_flatten.sum() / (original_label_flatten.sum() + 1e-7) 

    if include_mask.sum() == 0: 
        risk = 0 
    else:
        risk = 1 - IoU
    return tp.sum(), fp.sum(), fn.sum(), tn.sum(), risk, coverage    