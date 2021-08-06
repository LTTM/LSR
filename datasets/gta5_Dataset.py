# -*- coding: utf-8 -*-
from PIL import Image, ImageFile
import numpy as np
import os
import torch.utils.data as data
import imageio

from datasets.cityscapes_Dataset import City_Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
DEBUG = False

class GTA5_Dataset(City_Dataset):
    def __init__(self,
                 args,
                 data_root_path='./datasets/GTA5',
                 list_path='./datasets/GTA5',
                 split='train',
                 base_size=769,
                 crop_size=769,
                 training=True,
                 is_source=False,
                 use_weights=True):

        # setup attributes
        self.args = args
        self.data_path=data_root_path
        self.list_path=list_path
        self.split=split
        self.base_size = base_size if isinstance(base_size, tuple) else (base_size, base_size)
        self.crop_size = crop_size if isinstance(crop_size, tuple) else (crop_size, crop_size)
        self.training = training
        self.is_source = is_source

        # compute the lower limit for the rescaling process
        # relevant only when using random rescaling
        self.min_ratio = min(self.crop_size[0]/self.base_size[0], self.crop_size[1]/self.base_size[1]) # round to 3 decimal digits by excess
        self.min_ratio = max(self.min_ratio, 0.5)

        self.random_mirror = args.random_mirror
        self.random_crop = args.random_crop
        self.resize = args.resize
        self.gaussian_blur = args.gaussian_blur
        
        # wether to use the weighted version of histogram downsampling
        self.use_weights = self.args.weighted_hist


        item_list_filepath = os.path.join(self.list_path, self.split + ".txt")
        self.items = [id for id in open(item_list_filepath)]
        
        ignore_label = -1
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        
        self.class_16 = False
        self.class_13 = False


class GTA5_DataLoader():
    def __init__(self, args, training=True, is_source=False):

        self.args = args

        data_set = GTA5_Dataset(args,
                                data_root_path=args.data_root_path,
                                list_path=args.list_path,
                                split=args.split,
                                base_size=args.base_size,
                                crop_size=args.crop_size,
                                training=training,
                                is_source=is_source)

        if self.args.split == "train" or self.args.split == "trainval" or self.args.split =="all":
            self.data_loader = data.DataLoader(data_set,
                                batch_size=self.args.batch_size,
                                shuffle=True,
                                num_workers=self.args.data_loader_workers,
                                pin_memory=self.args.pin_memory,
                                drop_last=True)
        elif self.args.split =="val" or self.args.split == "test":
            self.data_loader = data.DataLoader(data_set,
                                batch_size=self.args.batch_size,
                                shuffle=False,
                                num_workers=self.args.data_loader_workers,
                                pin_memory=self.args.pin_memory,
                                drop_last=True)
        else:
            raise Warning("split must be train/val/trainavl/test/all")

        val_split = 'val' if self.args.split == "train" else 'test'
        val_set = GTA5_Dataset(args,
                                data_root_path=args.data_root_path,
                                list_path=args.list_path,
                                split=val_split,
                                base_size=args.base_size,
                                crop_size=args.crop_size,
                                training=False,
                                is_source=False)
        self.val_loader = data.DataLoader(val_set,
                                batch_size=self.args.batch_size,
                                shuffle=False,
                                num_workers=self.args.data_loader_workers,
                                pin_memory=self.args.pin_memory,
                                drop_last=True)
        self.valid_iterations = (len(val_set) + self.args.batch_size) // self.args.batch_size
        self.num_iterations = (len(data_set) + self.args.batch_size) // self.args.batch_size
