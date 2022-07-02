# -*- coding: utf-8 -*-
import random
from PIL import Image, ImageOps, ImageFilter, ImageFile
import numpy as np
import os
import torch
import torch.utils.data as data
import torchvision.transforms as ttransforms
import imageio
import time
import glob
import sys

from datasets.histDown import histDown

# fix truncated image error
ImageFile.LOAD_TRUNCATED_IMAGES = True


# GLOBAL VARIABLES DEFINITION
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

name_classes = [
    'road',
    'sidewalk',
    'building',
    'wall',
    'fence',
    'pole',
    'trafflight',
    'traffsign',
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
    'unlabeled'
]

# colour map
label_colours_19 = [
        # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
        [  0,   0,   0]] # the color of ignored label(-1)
label_colours_19 = list(map(tuple, label_colours_19))


# [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]
# colour map
label_colours_16 = [
        # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 60, 100],
        [0, 0, 230],
        [119, 11, 32],
        [  0,   0,   0]] # the color of ignored label(-1)
label_colours_16 = list(map(tuple, label_colours_16))

# [0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]
# colour map
label_colours_13 = [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 60, 100],
        [0, 0, 230],
        [119, 11, 32],
        [  0,   0,   0]] # the color of ignored label(-1)
label_colours_13 = list(map(tuple, label_colours_13))



"""

    Parent Class for all Datasets, here is defined the processing.

"""
class City_Dataset(data.Dataset):
    def __init__(self,
                 args,
                 data_root_path=os.path.abspath('./datasets/Cityscapes'),
                 list_path=os.path.abspath('./datasets/Cityscapes'),
                 split='train',
                 base_size=769,
                 crop_size=769,
                 training=True,
                 class_16=False,
                 class_13=False,
                 is_source=False,
                 use_weights=False):

        
        # setup attributes
        self.args = args
        self.data_path=data_root_path
        self.list_path=list_path
        self.split=split
        self.base_size = base_size if isinstance(base_size, tuple) else (base_size, base_size)
        self.crop_size = crop_size if isinstance(crop_size, tuple) else (crop_size, crop_size)
        self.training = training
        self.is_source = is_source
        
        # weighted histogram is available only for the source setup (13 classes)
        self.use_weights = self.args.weighted_hist and class_13

        # compute the lower limit for the rescaling process
        # relevant only when using random rescaling
        self.min_ratio = min(self.crop_size[0]/self.base_size[0], self.crop_size[1]/self.base_size[1]) # round to 3 decimal digits by excess
        self.min_ratio = max(self.min_ratio, 0.5)

        self.random_mirror = args.random_mirror
        self.random_crop = args.random_crop
        self.resize = args.resize
        self.gaussian_blur = args.gaussian_blur

        item_list_filepath = os.path.join(self.list_path, self.split + ".txt")

        try:
            time.sleep(2)
            self.items = [id for id in open(item_list_filepath)]
        except FileNotFoundError:
            print('sys.argv', os.path.dirname(sys.argv[0]))
            print('FileNotFoundError: cwdir is ' + os.getcwd())
            print(os.listdir(os.getcwd()))
            print('FileNotFoundError: parent cwdir is ' + os.path.dirname(os.getcwd()))
            print('glob:', glob.glob(os.path.dirname(os.getcwd())+ '/*'))
            print('os:', os.listdir(os.path.dirname(os.getcwd())))
            print('FileNotFoundError: parent of parent cwdir is ' + os.path.dirname(os.path.dirname(os.getcwd())))
            print(os.listdir(os.path.dirname(os.path.dirname(os.getcwd()))))
            self.items = [id for id in open(item_list_filepath)]

        ignore_label = -1

        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}


        # SYNTHIA -> CityScapes Adaptation
        self.class_16 = class_16
        synthia_set_16 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]
        self.trainid_to_16id = {id:i for i,id in enumerate(synthia_set_16)}
        self.trainid_to_16id[255] = ignore_label

        # CityScapes -> CrossCity Adaptation
        self.class_13 = class_13
        synthia_set_13 = [0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]
        self.trainid_to_13id = {id:i for i,id in enumerate(synthia_set_13)}

    # maps original labels into train IDs
    def id2trainId(self, label, reverse=False, ignore_label=-1):
        label_copy = ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        if self.class_16:
            label_copy_16 = ignore_label * np.ones(label.shape, dtype=np.float32)
            for k, v in self.trainid_to_16id.items():
                label_copy_16[label_copy == k] = v
            label_copy = label_copy_16
        if self.class_13:
            label_copy_13 = ignore_label * np.ones(label.shape, dtype=np.float32)
            for k, v in self.trainid_to_13id.items():
                label_copy_13[label_copy == k] = v
            label_copy = label_copy_13
        return label_copy

    # gets each sample
    def __getitem__(self, item):
        id_img, id_gt = self.items[item].strip('\n').split(' ')
        image_path = self.data_path + id_img
        image = Image.open(image_path).convert("RGB")
        
        gt_image_path = self.data_path + id_gt
        gt_image = Image.open(gt_image_path)

        if (self.split == "train" or self.split == "trainval") and self.training:
            image, gt_image, gt_down = self._train_sync_transform(image, gt_image)
        else:
            image, gt_image, gt_down = self._val_sync_transform(image, gt_image)

        return image, gt_image, gt_down, item

    def _train_sync_transform(self, img, mask):
        # random mirroring, defaults to true
        if self.random_mirror and random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if mask: mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # wether images are resized or kept at original resolution, defaults to true
        if self.resize:
            if self.args.random_resize: # wether images are randomly resized and cropped, defaults to false
                ratio = random.random()*(1.-self.min_ratio) + self.min_ratio # random scale between min_ratio and 1
                size = (int(self.base_size[0]*ratio), int(self.base_size[1]*ratio))
            else:
                size = self.base_size

            img = img.resize(size, Image.BICUBIC)
            if mask: mask = mask.resize(size, Image.NEAREST)

        size = img.size

        # wether to extract image patches of crop_size, defaults to false.
        # necessary when using random resize
        if self.random_crop or self.args.random_resize:
            dw, dh = size[0]-self.crop_size[0], size[1]-self.crop_size[1]
            ow = random.randrange(0,dw) if dw > 0 else 0
            oh = random.randrange(0,dh) if dh > 0 else 0

            img = img.crop((ow,oh,ow+self.crop_size[0],oh+self.crop_size[1]))
            if mask: mask = mask.crop((ow,oh,ow+self.crop_size[0],oh+self.crop_size[1]))

        # wether to use color jittering ( only if dataset is source ), defaults to true
        if self.is_source and self.args.color_jitter:
            # color jittering
            # all random variables have as mean the original value, white: 255, delta: 0
            image = np.asarray(img, np.float32)
            
            # white re-balancing (multiplicative jitter)
            new_red = 255./random.randint(180,330)
            new_green = 255./random.randint(180,330)
            new_blue = 255./random.randint(180,330)
            
            # color shift (additive jitter)
            delta_red = random.randint(-20,20)
            delta_green = random.randint(-20,20)
            delta_blue = random.randint(-20,20)
            
            new_white = np.array([new_red, new_green, new_blue])
            delta_colors = np.array([delta_red, delta_green, delta_blue])
            image = np.clip(image*new_white+delta_colors, 0, 255)

            # convert back to PIL image
            img = Image.fromarray(np.uint8(image))

        # wether to apply random gaussian blurring, defaults to true
        if self.gaussian_blur and random.random() < 0.5:
                img = img.filter( ImageFilter.GaussianBlur(radius=random.random()) )
        
        # apply shared transformations and convert to tensor
        if mask:
            img, (mask, mask_down) = self._img_transform(img, train=True), self._mask_transform(mask)
            return img, mask, mask_down
        else:
            img = self._img_transform(img, train=True)
            return img

    # in validation we do not apply any data augmentation
    def _val_sync_transform(self, img, mask):
        if self.resize:
            img = img.resize(self.base_size, Image.BICUBIC)
            #if not self.args.fullres_labels:
            mask = mask.resize(self.base_size, Image.NEAREST)

        # apply shared transformations and convert to tensor
        img, (mask, mask_down) = self._img_transform(img), self._mask_transform(mask)
        return img, mask, mask_down

    # shared transformations and tensor conversion
    def _img_transform(self, image, train=False):
        # wether to use numpy or pytorch for the transformations
        # defaults to numpy (flag=true)
        if self.args.numpy_transform:
            image = np.asarray(image, np.float32)
            image = image[:, :, ::-1]  # change to BGR
            image -= IMG_MEAN # subtract global color mean, as defined in the globals
            image = image.transpose((2, 0, 1)).copy() # (C x H x W)
            new_image = torch.from_numpy(image)
        else:
            image_transforms = ttransforms.Compose([
                ttransforms.ToTensor(),
                ttransforms.Normalize([.485, .456, .406], [.229, .224, .225]),
            ])
            new_image = image_transforms(image)
        return new_image

    # shared mask transformations, computes output and feature level labels
    def _mask_transform(self, gt_image):
    
        target = np.asarray(gt_image, np.float32)
        target = self.id2trainId(target).copy()

        # resnet networks have broken padding, we need to add 1px in both
        # directions to have a matching map
        if self.args.backbone == 'resnet101' or self.args.backbone == 'resnet50':
            shape = (target.shape[1]//8+1, target.shape[0]//8+1)
        else:
            shape = (target.shape[1]//8, target.shape[0]//8)
        
        # if downsampling is histogram-aware use the appropriate function, defaults to true
        if self.args.down_type == 'hist':
            mask_down = torch.LongTensor(histDown(target.astype(int), shape, self.args.hist_th, self.args.num_classes, use_weights=self.use_weights))
        else:
            im = Image.fromarray(target).resize(shape, Image.NEAREST)
            mask_down = torch.LongTensor(np.array(im))

        target = torch.LongTensor(target)
        return target, mask_down

    def __len__(self):
        return len(self.items)


# dataloader, contains both train and validation (or validation and test) datasets
class City_DataLoader():
    def __init__(self, args, training=True, is_source=False):

        self.args = args

        data_set = City_Dataset(args,
                                data_root_path=self.args.data_root_path,
                                list_path=self.args.list_path,
                                split=args.split,
                                base_size=args.base_size,
                                crop_size=args.crop_size,
                                training=training,
                                class_16=args.class_16,
                                class_13=args.class_13,
                                is_source=is_source)

        if (self.args.split == "train" or self.args.split == "trainval") and training:
            self.data_loader = data.DataLoader(data_set,
                                batch_size=self.args.batch_size,
                                shuffle=True,
                                num_workers=self.args.data_loader_workers,
                                pin_memory=self.args.pin_memory,
                                drop_last=True)
        else:
            self.data_loader = data.DataLoader(data_set,
                                batch_size=self.args.batch_size,
                                shuffle=False,
                                num_workers=self.args.data_loader_workers,
                                pin_memory=self.args.pin_memory,
                                drop_last=True)

        val_set = City_Dataset(args,
                                data_root_path=self.args.data_root_path,
                                list_path=self.args.list_path,
                                split='test' if self.args.split == "test" else 'val',
                                base_size=args.base_size,
                                crop_size=args.crop_size,
                                training=False,
                                class_16=args.class_16,
                                class_13=args.class_13,
                                is_source=is_source)
        self.val_loader = data.DataLoader(val_set,
                                batch_size=self.args.batch_size,
                                shuffle=False,
                                num_workers=self.args.data_loader_workers,
                                pin_memory=self.args.pin_memory,
                                drop_last=True)
                                
        self.valid_iterations = (len(val_set) + self.args.batch_size) // self.args.batch_size
        self.num_iterations = (len(data_set) + self.args.batch_size) // self.args.batch_size


def flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    inds = tuple(slice(None, None) if i != dim
             else x.new(torch.arange(x.size(i)-1, -1, -1).tolist()).long()
             for i in range(x.dim()))
    return x[inds]


def inv_preprocess(imgs, num_images=1, img_mean=IMG_MEAN, numpy_transform=False):
    """Inverse preprocessing of the batch of images.

    Args:
      imgs: batch of input images.
      num_images: number of images to apply the inverse transformations on.
      img_mean: vector of mean colour values.
      numpy_transform: whether change RGB to BGR during img_transform.

    Returns:
      The batch of the size num_images with the same spatial dimensions as the input.
    """
    if numpy_transform:
        imgs = flip(imgs, 1)
    def norm_ip(img, min, max):
        img.clamp_(min=min, max=max)
        img.add_(-min).div_(max - min + 1e-5)
    norm_ip(imgs, float(imgs.min()), float(imgs.max()))
    return imgs

def decode_labels(mask, num_classes, num_images=1):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict.

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    assert num_classes in [13, 16, 19]
    label_colours = label_colours_16 if num_classes==16 else label_colours_13 if num_classes==13 else label_colours_19

    if isinstance(mask, torch.Tensor):
        mask = mask.data.cpu().numpy()
    n, h, w = mask.shape
    if n < num_images:
        num_images = n
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
      img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
      pixels = img.load()
      for j_, j in enumerate(mask[i, :, :]):
          for k_, k in enumerate(j):
              if k < num_classes:
                  pixels[k_,j_] = label_colours[k]
      outputs[i] = np.array(img)
    return torch.from_numpy(outputs.transpose([0, 3, 1, 2]).astype('float32')).div_(255.0)
