# -*- coding: utf-8 -*-
import random
import scipy.io
from PIL import Image, ImageOps, ImageFilter, ImageFile
import numpy as np
import copy
import os
import torch
import torch.utils.data as data
import torchvision.transforms as ttransforms

from datasets.cityscapes_Dataset import City_Dataset, City_DataLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True

class CrossCity_Dataset(City_Dataset):
    def __init__(self,
                 args,
                 data_root_path='./datasets/NTHU_Datasets/Rio',
                 list_path='./datasets/NTHU_list/Rio/List',
                 split='train',
                 base_size=769,
                 crop_size=769,
                 training=True,
                 class_13=False,
                 is_source=False):

        # setup attributes
        self.args = args
        self.data_path=data_root_path
        self.list_path=list_path
        self.split=split
        self.base_size = base_size if isinstance(base_size, tuple) else (base_size, base_size)
        self.crop_size = crop_size if isinstance(crop_size, tuple) else (crop_size, crop_size)
        self.training = training
        self.is_source = is_source
        
        # crosscity is never a source dataset!
        self.use_weights = False

        # compute the lower limit for the rescaling process
        # relevant only when using random rescaling
        self.min_ratio = min(self.crop_size[0]/self.base_size[0], self.crop_size[1]/self.base_size[1]) # round to 3 decimal digits by excess
        self.min_ratio = max(self.min_ratio, 0.5)

        self.random_mirror = args.random_mirror
        self.random_crop = args.random_crop
        self.resize = args.resize
        self.gaussian_blur = args.gaussian_blur

        if self.split == 'train':
            item_list_filepath = os.path.join(self.list_path, "train.txt")
            self.image_filepath = os.path.join(self.data_path, "Images/Train")
            self.gt_filepath = os.path.join(self.data_path, "Labels/Train")
        elif self.split == 'val':
            item_list_filepath = os.path.join(self.list_path, "test.txt")
            self.image_filepath = os.path.join(self.data_path, "Images/Test")
            self.gt_filepath = os.path.join(self.data_path, "Labels/Test")
        else:
            raise Warning("split must be train/val")

        self.items = [id.strip() for id in open(item_list_filepath)]

        ignore_label = -1
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        self.class_16 = False
        # only consider 13 shared classes
        self.class_13 = self.args.class_13 or class_13
        synthia_set_13 = [0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]
        self.trainid_to_13id = {id:i for i,id in enumerate(synthia_set_13)}

        print("{} num images in City {} set have been loaded.".format(len(self.items), self.split))

    # override get item, during training we don't have labels
    def __getitem__(self, item):
        id = self.items[item]

        image_path = os.path.join(self.image_filepath, "{}.jpg".format(id))
        if not os.path.exists(image_path):
            image_path = os.path.join(self.image_filepath, "{}.png".format(id))
        image = Image.open(image_path).convert("RGB")

        if self.split == "train" and self.training:
            image = self._train_sync_transform(image, None)
            return image, image, image, item
        else:
            gt_image_path = os.path.join(self.gt_filepath, "{}_eval.png".format(id))
            gt_image = Image.open(gt_image_path)
            image, gt_image, gt_down = self._val_sync_transform(image, gt_image)

        return image, gt_image, gt_down, item

class CrossCity_DataLoader():
    def __init__(self, args, training=True, **kwargs):

        self.args = args

        data_set = CrossCity_Dataset(args, 
                                data_root_path=args.data_root_path,
                                list_path=args.list_path,
                                split=args.split,
                                base_size=args.base_size,
                                crop_size=args.crop_size,
                                training=training)
        self.len = len(data_set)

        if self.args.split == "train":
            self.data_loader = data.DataLoader(data_set,
                                batch_size=self.args.batch_size,
                                shuffle=True,
                                num_workers=self.args.data_loader_workers,
                                pin_memory=self.args.pin_memory,
                                drop_last=True)
        else:
            raise Warning("split must be train")

        val_split = 'val'
        val_set = CrossCity_Dataset(args, 
                                data_root_path=args.data_root_path,
                                list_path=args.list_path,
                                split=val_split,
                                base_size=args.base_size,
                                crop_size=args.crop_size,
                                training=False)
        self.val_loader = data.DataLoader(val_set,
                                batch_size=self.args.batch_size,
                                shuffle=False,
                                num_workers=self.args.data_loader_workers,
                                pin_memory=self.args.pin_memory,
                                drop_last=True)
                                
        self.valid_iterations = (len(val_set) + self.args.batch_size) // self.args.batch_size
        self.num_iterations = (len(data_set) + self.args.batch_size) // self.args.batch_size
    
    def __len__(self):
        return self.len