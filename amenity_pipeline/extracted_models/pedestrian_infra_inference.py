import torch
import os
from collections import namedtuple
import torch.nn as nn
import math
import importlib
from torch.utils import data
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as standard_transforms
from torchvision import transforms as standard_transforms
import transforms as extended_transforms
import numpy as np  
from PIL import Image  
from collections import OrderedDict
from collections import OrderedDict
import time  
from pedestrian_infra_utils import Loader, forgiving_state_restore, flip_tensor, dihedral_transform, NumpyToTensor, poly_schd
from pedestrian_infra_model import ImageBasedCrossEntropyLoss2d, HRNet_Mscale

crop_size = [1024,1024]
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# torch.cuda.set_device(0)
net: torch.nn.parallel.DataParallel
val_loader: DataLoader



val_input_transform = standard_transforms.Compose([
        NumpyToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
target_transform = extended_transforms.MaskToTensor()
a = 0


def predict(pixelBlocks,net, device):
    import pickle as pkl
    
    import datetime

    flips = [1, 0]
    output = 0.
    for flip in flips:
        with torch.no_grad():
            img = pixelBlocks['raster_pixels']
            if flip == 1:
                img = flip_tensor(img, 0)
            transformed_image = val_input_transform(img)
            transformed_image = transformed_image.unsqueeze(0)  

            
            inputs = {'images': transformed_image, 'gts': torch.randn(1, 3, 1024, 1024)}
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            output_dict = net(inputs)
            _pred = output_dict['pred']
            if flip == 1:
                output = output + flip_tensor(_pred, 0)
            else:
                output = output + _pred
    output = output / len(flips)
    output_data = torch.nn.functional.softmax(output, dim=1).cpu().data
    max_probs, predictions = output_data.max(1)
    predictions = predictions.numpy()
    return predictions



def predict_tta(pixelBlocks,net, device):
    
    flips = [0]
    output = 0.
    all_predictions = []
    trans = [0,1,2,3,4,5,6,7]
    for k in trans:
        with torch.no_grad():
            img = pixelBlocks['raster_pixels']
            transformed_image = val_input_transform(img)
            transformed_image = transformed_image.unsqueeze(0)
            flipped_image_tensor = dihedral_transform(transformed_image, k)
            print(flipped_image_tensor.shape)
            if k in [4,5, 6,7]:
                flipped_image_tensor = flipped_image_tensor.permute(0, 2, 1, 3) 
            
            inputs = {'images': flipped_image_tensor, 'gts': torch.randn(1, 3, 1024, 1024)}
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            output_dict = net(inputs)
            _pred = output_dict['pred']
            corrected_prediction = dihedral_transform(_pred, k)
            if k in [4,5,6,7]:
                corrected_prediction = dihedral_transform(_pred, k).permute(0, 2, 1, 3) 
                
            all_predictions.append(corrected_prediction)
    all_predictions = torch.stack(all_predictions)
    predictions = all_predictions.mean(dim=0, keepdim=True)
    output_data = torch.nn.functional.softmax(predictions[0], dim=1).cpu().data
    max_probs, predictions = output_data.max(1)
    predictions = predictions.numpy()
    return predictions
