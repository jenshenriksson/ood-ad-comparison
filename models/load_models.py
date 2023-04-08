import torch 
import torch.nn as nn

import models.deeplabv3
from models.pspnet import PSPNet

def deeplab_mobilenet(num_classes): 
    model = models.deeplabv3.modeling.__dict__['deeplabv3plus_mobilenet'](num_classes=num_classes)
    PATH_TO_PTH = './models/pretrained/best_deeplabv3plus_mobilenet_cityscapes_os16.pth'
    model.load_state_dict( torch.load( PATH_TO_PTH )['model_state'])
    return model

def deeplab_resnet101(num_classes): 
    model = models.deeplabv3.modeling.__dict__['deeplabv3plus_resnet101'](num_classes=num_classes)
    PATH_TO_PTH = './models/pretrained/best_deeplabv3plus_resnet101_cityscapes_os16.pth'
    model.load_state_dict( torch.load( PATH_TO_PTH )['model_state']  )
    return model 

def pspnet(num_classes): 
    layers = 101
    zoom_factor = 8
    loss = nn.CrossEntropyLoss()
    pspnet_model = PSPNet(layers=layers, classes=num_classes, zoom_factor=zoom_factor, criterion=loss, pretrained=False)

    # Due to training was done in an older version of Torch, we need to remap training to CPU
    # and hope we can re-assign the weight to the layers ... 
    checkpoint = torch.load('./models/pretrained/pspnet_train_epoch_200.pth', map_location=lambda storage, loc: storage.cuda())
    # 
    state_dict = checkpoint['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
        
    pspnet_model.load_state_dict(new_state_dict)
    return pspnet_model
