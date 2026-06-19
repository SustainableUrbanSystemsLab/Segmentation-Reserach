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
import transforms as extended_transforms
import numpy as np  
from PIL import Image  
from collections import OrderedDict
import time  


"""Source License
# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
"""

class NumpyToTensor(object):  
    def __call__(self, numpy_array):  
        # Convert numpy array to a torch tensor  
        tensor = torch.from_numpy(numpy_array)  
        # If the input is a numpy array in HWC format (Height, Width, Channels),  
        # you need to permute it to CHW format (Channels, Height, Width)  
        if len(tensor.shape) == 3 and tensor.shape[2] == 3:  
            tensor = tensor.permute(2, 0, 1)  
        return tensor.float() / 255 # To match the ToTensor() behavior  

class AttrDict(dict):

    IMMUTABLE = '__immutable__'

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__[AttrDict.IMMUTABLE] = False

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if not self.__dict__[AttrDict.IMMUTABLE]:
            if name in self.__dict__:
                self.__dict__[name] = value
            else:
                self[name] = value
        else:
            raise AttributeError(
                'Attempted to set "{}" to "{}", but AttrDict is immutable'.
                format(name, value)
            )

    def immutable(self, is_immutable):
        """Set immutability to is_immutable and recursively apply the setting
        to all nested AttrDicts.
        """
        self.__dict__[AttrDict.IMMUTABLE] = is_immutable
        # Recursively set immutable state
        for v in self.__dict__.values():
            if isinstance(v, AttrDict):
                v.immutable(is_immutable)
        for v in self.values():
            if isinstance(v, AttrDict):
                v.immutable(is_immutable)

    def is_immutable(self):
        return self.__dict__[AttrDict.IMMUTABLE]


def poly_schd(epoch):
        return math.pow(1 - epoch / 300, 2)
        
def Norm2d(in_channels, **kwargs):
    """
    Custom Norm Function to allow flexible switching
    """
    #layer = getattr(AttrDict(), 'BNFUNC')
    layer = torch.nn.BatchNorm2d
    normalization_layer = layer(in_channels, **kwargs)
    return normalization_layer

def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear',
                                     align_corners=False)
                                     
def ResizeX(x, scale_factor):
    '''
    scale x by some factor
    '''
    x_scaled = torch.nn.functional.interpolate(
        x, scale_factor=scale_factor, mode='bilinear',
        align_corners=False, recompute_scale_factor=True)
    
    return x_scaled

def scale_as(x, y):
    '''
    scale x to the same size as y
    '''
    y_size = y.size(2), y.size(3)
    x_scaled = torch.nn.functional.interpolate(
        x, size=y_size, mode='bilinear',
        align_corners=False)
    return x_scaled

def random_sampling(alist, num):
    """
    Randomly sample num items from the list
    alist: list of centroids to sample from
    num: can be larger than the list and if so, then wrap around
    return: class uniform samples from the list
    """
    sampling = []
    len_list = len(alist)
    assert len_list, 'len_list is zero!'
    indices = np.arange(len_list)
    np.random.shuffle(indices)

    for i in range(num):
        item = alist[indices[i % len_list]]
        sampling.append(item)
    return sampling

def dihedral_transform(x, k):  # expects [C, H, W]
    flips = []
    if k & 1:
        flips.append(1)
    if k & 2:
        flips.append(2)
    if flips:
        x = torch.flip(x, flips)
    if k & 4:
        x = x.transpose(1, 2)
    return x.contiguous()

def flip_tensor(x, dim):
    """
    Flip Tensor along a dimension
    """
    #dim = x.dim() + dim if dim < 0 else dim
    #return x[tuple(slice(None, None) if i != dim
    #               else torch.arange(x.size(i) - 1, -1, -1).long()
    #               for i in range(x.dim()))]
    dim = x.ndim + dim if dim < 0 else dim  
    return x[tuple(slice(None, None) if i != dim  
                   else np.arange(x.shape[i] - 1, -1, -1)  
                   for i in range(x.ndim))]  

def build_epoch(imgs, centroids, num_classes, train):
    """
    Generate an epoch of crops using uniform sampling.
    Needs to be called every epoch.
    Will not apply uniform sampling if not train or class uniform is off.

    Inputs:
      imgs - list of imgs
      centroids - list of class centroids
      num_classes - number of classes
      class_uniform_pct: % of uniform images in one epoch
    Outputs:
      imgs - list of images to use this epoch
    """
    class_uniform_pct = 0.5
    if not (train and class_uniform_pct):
        return imgs

    #logger.debug("Class Uniform Percentage: {}".format(str(class_uniform_pct)))
    num_epoch = int(len(imgs))

    #logger.debug('Class Uniform items per Epoch: {}'.format(str(num_epoch)))
    num_per_class = int((num_epoch * class_uniform_pct) / num_classes)
    class_uniform_count = num_per_class * num_classes
    num_rand = num_epoch - class_uniform_count
    # create random crops
    imgs_uniform = random_sampling(imgs, num_rand)

    # now add uniform sampling
    for class_id in range(num_classes):
        msg = "cls {} len {}".format(class_id, len(centroids[class_id]))
        #logger.debug(msg)
    for class_id in range(num_classes):
        
        num_per_class_biased = num_per_class
        centroid_len = len(centroids[class_id])
        if centroid_len == 0:
            pass
        else:
            class_centroids = random_sampling(centroids[class_id],
                                              num_per_class_biased)
            imgs_uniform.extend(class_centroids)

    return imgs_uniform

class BaseLoader(data.Dataset):
    def __init__(self, quality, mode, joint_transform_list, img_transform,
                 label_transform):

        super(BaseLoader, self).__init__()
        self.quality = quality
        self.mode = mode
        self.joint_transform_list = joint_transform_list
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.train = mode == 'train'
        self.id_to_trainid = {}
        self.centroids = None
        self.all_imgs = None
        self.drop_mask = np.zeros((1024, 2048))
        self.drop_mask[15:840, 14:2030] = 1.0

    def build_epoch(self):
        """
        For class uniform sampling ... every epoch, we want to recompute
        which tiles from which images we want to sample from, so that the
        sampling is uniformly random.
        """
        self.imgs = build_epoch(self.all_imgs,
                                        self.centroids,
                                        self.num_classes,
                                        self.train)

    @staticmethod
    def find_images(img_root, mask_root, img_ext, mask_ext):
        """
        Find image and segmentation mask files and return a list of
        tuples of them.
        """
        img_path = '{}/*.{}'.format(img_root, img_ext)
        imgs = glob.glob(img_path)
        items = []
        for full_img_fn in imgs:
            img_dir, img_fn = os.path.split(full_img_fn)
            img_name, _ = os.path.splitext(img_fn)
            full_mask_fn = '{}.{}'.format(img_name, mask_ext)
            full_mask_fn = os.path.join(mask_root, full_mask_fn)
            assert os.path.exists(full_mask_fn)
            items.append((full_img_fn, full_mask_fn))
        return items

    def disable_coarse(self):
        pass

    def colorize_mask(self, image_array):
        """
        Colorize the segmentation mask
        """
        new_mask = Image.fromarray(image_array.astype(np.uint8)).convert('P')
        new_mask.putpalette(self.color_mapping)
        return new_mask

    def dump_images(self, img_name, mask, centroid, class_id, img):
        img = tensor_to_pil(img)
        outdir = 'new_dump_imgs_{}'.format(self.mode)
        os.makedirs(outdir, exist_ok=True)
        if centroid is not None:
            dump_img_name = '{}_{}'.format(self.trainid_to_name[class_id],
                                           img_name)
        else:
            dump_img_name = img_name
        out_img_fn = os.path.join(outdir, dump_img_name + '.png')
        out_msk_fn = os.path.join(outdir, dump_img_name + '_mask.png')
        out_raw_fn = os.path.join(outdir, dump_img_name + '_mask_raw.png')
        mask_img = self.colorize_mask(np.array(mask))
        raw_img = Image.fromarray(np.array(mask))
        img.save(out_img_fn)
        mask_img.save(out_msk_fn)
        raw_img.save(out_raw_fn)

    def do_transforms(self, img, mask, centroid, img_name, class_id):
        """
        Do transformations to image and mask

        :returns: image, mask
        """
        scale_float = 1.0
        if self.joint_transform_list is not None:
            for idx, xform in enumerate(self.joint_transform_list):
                if idx == 0 and centroid is not None:
                    # HACK! Assume the first transform accepts a centroid
                    outputs = xform(img, mask, centroid)
                else:
                    outputs = xform(img, mask)

                if len(outputs) == 3:
                    img, mask, scale_float = outputs
                else:
                    img, mask = outputs
        print(self.img_transform)
        if self.img_transform is not None:
            img = self.img_transform(img)


        if self.label_transform is not None:
            mask = self.label_transform(mask)

        return img, mask, scale_float

    def read_images(self, img_path, mask_path):
        img = Image.open(img_path).convert('RGB')
        if mask_path is None or mask_path == '':
            w, h = img.size
            mask = np.zeros((h, w))
        else:
            mask = Image.open(mask_path)

        img_name = os.path.splitext(os.path.basename(img_path))[0]

        mask = np.array(mask)
        mask = mask.copy()
        for k, v in self.id_to_trainid.items():
            binary_mask = (mask == k)
            mask[binary_mask] = v

        mask = Image.fromarray(mask.astype(np.uint8))
        return img, mask, img_name

    def __getitem__(self, index):
        """
        Generate data:

        :return:
        - image: image, tensor
        - mask: mask, tensor
        - image_name: basename of file, string
        """
        # Pick an image, fill in defaults if not using class uniform
        if len(self.imgs[index]) == 2:
            img_path, mask_path = self.imgs[index]
            centroid = None
            class_id = None
        else:
            img_path, mask_path, centroid, class_id = self.imgs[index]

        img, mask, img_name = self.read_images(img_path, mask_path)
        
        if 'refinement' in mask_path:
            mask = np.array(mask)
            prob_mask_path = mask_path.replace('.png', '_prob.png')
            # put it in 0 to 1
            prob_map = np.array(Image.open(prob_mask_path)) / 255.0
            prob_map_threshold = (prob_map < 0.5)
            mask[prob_map_threshold] = -1
            mask = Image.fromarray(mask.astype(np.uint8))

        img, mask, scale_float = self.do_transforms(img, mask, centroid,
                                                    img_name, class_id)
        return img, mask, img_name, scale_float

    def __len__(self):
        return len(self.imgs)

    def calculate_weights(self):
        raise BaseException("not supported yet")

def make_dataset_folder(folder, testing=None):
    """
    Create Filename list for images in the provided path

    input: path to directory with *only* images files
	   test_mode: for test only with no ground truth 
   
    returns: items list with None filled for mask path
    """
    items = os.listdir(folder)
    if testing:
                
        items = [(os.path.join(folder, f), '') for f in items]
    else:
        mask_root = folder.replace('images', 'annotations', 6)
        items = [(os.path.join(folder, f), os.path.join(mask_root, f)) for f in items]
    
    items = sorted(items)

    """
    orig_len = len(items)
    rem = orig_len % 8
    if rem != 0:
        items = items[:-rem]

    msg = 'Found {} folder imgs but altered to {} to be modulo-8'
    msg = msg.format(orig_len, len(items))
    print(msg)
    """

    return items

def build_centroids(imgs, num_classes, train, cv=None, coarse=False,
                    custom_coarse=False, id2trainid=None):
    """
    The first step of uniform sampling is to decide sampling centers.
    The idea is to divide each image into tiles and within each tile,
    we compute a centroid for each class to indicate roughly where to
    sample a crop during training.

    This function computes these centroids and returns a list of them.
    """
    if not (0.5 and train):
        return []

    centroid_fn = ''
    
    if coarse or custom_coarse:
        if coarse:
            centroid_fn += '_coarse'
        if custom_coarse:
            centroid_fn += '_customcoarse_final'
    else:
        centroid_fn += '_cv{}'.format(cv)
    centroid_fn += '_tile{}.json'.format(512)
    json_fn = os.path.join(None,
                           centroid_fn)
    if os.path.isfile(json_fn):
        #logger.debug('Loading centroid file {}'.format(json_fn))
        with open(json_fn, 'r') as json_data:
            centroids = json.load(json_data)
        centroids = {int(idx): centroids[idx] for idx in centroids}
        #logger.debug('Found {} centroids'.format(len(centroids)))
    else:
        print('Didn\'t find {}, so building it'.format(json_fn))
        GLOBAL_RANK = 0
        if GLOBAL_RANK==0:

            os.makedirs('', exist_ok=True)
            # centroids is a dict (indexed by class) of lists of centroids
            print(imgs)
            centroids = class_centroids_all(
                imgs,
                num_classes,
                id2trainid=id2trainid)
            with open(json_fn, 'w') as outfile:
                json.dump(centroids, outfile, indent=4)

        # wait for everyone to be at the same point
        torch.distributed.barrier()

        #  GPUs (except rank0) read in the just-created centroid file
        if GLOBAL_RANK != 0:
            msg = f'Expected to find {json_fn}'
            assert os.path.isfile(json_fn), msg
            with open(json_fn, 'r') as json_data:
                centroids = json.load(json_data)
            centroids = {int(idx): centroids[idx] for idx in centroids}
        
    return centroids

class Loader(BaseLoader):
    num_classes = 4
    ignore_label = -1
    trainid_to_name = {}
    color_mapping = []

    def __init__(self, mode, quality='semantic', joint_transform_list=None,
                 img_transform=None, label_transform=None, eval_folder=None):

        BaseLoader.__init__(self,quality=quality,
                                     mode=mode,
                                     joint_transform_list=joint_transform_list,
                                     img_transform=img_transform,
                                     label_transform=label_transform)

        #root = cfg.DATASET.SATELLITE_DIR
        # config_fn = os.path.join(root, 'config.json')
        # self.fill_colormap_and_names(config_fn)
        Label = namedtuple( 'Label' , [
            
                'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                # We use them to uniquely name a class
            
                'id'          , # An integer ID that is associated with this label.
                # The IDs are used to represent the label in ground truth images
                # An ID of -1 means that this label does not have an ID and thus
                # is ignored when creating ground truth images (e.g. license plate).
                # Do not modify these IDs, since exactly these IDs are expected by the
                # evaluation server.
            
                'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
            # ground truth images with train IDs, using the tools provided in the
            # 'preparation' folder. However, make sure to validate or submit results
            # to our evaluation server using the regular IDs above!
            # For trainIds, multiple labels might have the same ID. Then, these labels
            # are mapped to the same class in the ground truth images. For the inverse
            # mapping, we use the label that is defined first in the list below.
            # For example, mapping all void-type classes to the same ID in training,
            # might make sense for some approaches.
            # Max value is 255!
                'color'       , # The color of this label
            ] )
        labels = [
            #       name                     id    trainId     color
            Label(  'Sidewalk'             ,  1 ,        0 ,  (0, 0, 255)),
            Label(  'Road'                 ,  2 ,        1 ,  (0, 128, 0)),
            Label(  'Crosswalk'            ,  3 ,        2 ,  (255, 0, 0)),
            Label(  'Background'           ,  4 ,        3 ,  (0, 0, 0)),
            ]
        label2trainid   = { label.id      : label.trainId for label in labels   }
        trainId2name   = { label.trainId : label.name for label in labels   }
        self.id_to_trainid = label2trainid
        self.trainid_to_name = trainId2name
        self.fill_colormap()

        ######################################################################
        # Assemble image lists
        ######################################################################
        
        
        self.all_imgs = make_dataset_folder(eval_folder, testing=True)

        # logger.debug('all imgs {}'.format(len(self.all_imgs)))
        #logger.debug('all imgs {}'.format(len(self.all_imgs)))
        self.fine_centroids = build_centroids(self.all_imgs,
                                                      self.num_classes,
                                                      self.train,
                                                      cv=0,
                                                      id2trainid=self.id_to_trainid)
        self.centroids = self.fine_centroids
#        print('centroids', self.centroids)
        self.build_epoch()


    def fill_colormap(self):
        palette = [0, 0, 255,
                   0, 128, 0,
                   255, 0, 0,
                   0, 0, 0]

        zero_pad = 256 * 3 - len(palette)
        for i in range(zero_pad):
            palette.append(0)
        self.color_mapping = palette

class MaskToTensor(object):
    def __call__(self, img, blockout_predefined_area=False):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()

def forgiving_state_restore(net, loaded_dict):
    """
    Handle partial loading when some tensors don't match up in size.
    Because we want to use models that were trained off a different
    number of classes.
    """

    net_state_dict = net.state_dict()
    new_loaded_dict = {}
    for k in net_state_dict:
        new_k = 'module.'+k
        if new_k in loaded_dict and net_state_dict[k].size() == loaded_dict[new_k].size():
            new_loaded_dict[k] = loaded_dict[new_k]
        else:            
            print("Skipped loading parameter {}".format(k))
    net_state_dict.update(new_loaded_dict)
    net.load_state_dict(net_state_dict)
    return net