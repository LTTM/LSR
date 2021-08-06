# -*- coding: utf-8 -*-
from PIL import Image
import numpy as np
import os
import torch.utils.data as data
import imageio

from datasets.cityscapes_Dataset import City_Dataset


DEBUG = False

imageio.plugins.freeimage.download()

class SYNTHIA_Dataset(City_Dataset):
    def __init__(self,
                 args,
                 data_root_path='./datasets/SYNTHIA',
                 list_path='./datasets/SYNTHIA',
                 split='train',
                 base_size=769,
                 crop_size=769,
                 training=True,
                 class_16=False,
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

        self.id_to_trainid = {1: 10, 2: 2, 3: 0, 4: 1, 5: 4, 6: 8, 7: 5, 8: 13,
                              9: 7, 10: 11, 11: 18, 12: 17, 15: 6, 16: 9, 17: 12,
                              18: 14, 19: 15, 20: 16, 21: 3}

        # only consider 16 shared classes
        self.class_16 = class_16
        synthia_set_16 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]
        self.trainid_to_16id = {id: i for i, id in enumerate(synthia_set_16)}
        self.class_13 = False
        

    # override method, synthia images are stored in a different format
    def __getitem__(self, item):
        
        id_img, id_gt = self.items[item].strip('\n').split(' ')
        image_path = self.data_path + id_img
        image = Image.open(image_path).convert("RGB")
        
        gt_image_path = self.data_path + id_gt
        gt_image = imageio.imread(gt_image_path, format='PNG-FI')[:, :, 0]
        gt_image = Image.fromarray(np.uint8(gt_image))
        
        if (self.split == "train" or self.split == "trainval" or self.split =="all") and self.training:
            image, gt_image, gt_down = self._train_sync_transform(image, gt_image)
        else:
            image, gt_image, gt_down = self._val_sync_transform(image, gt_image)

        return image, gt_image, gt_down, item

class SYNTHIA_DataLoader():
    def __init__(self, args, training=True, is_source=False):

        self.args = args

        data_set = SYNTHIA_Dataset(args,
                                data_root_path=args.data_root_path,
                                list_path=args.list_path,
                                split=args.split,
                                base_size=args.base_size,
                                crop_size=args.crop_size,
                                training=training,
                                class_16=args.class_16,
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
        val_set = SYNTHIA_Dataset(args,
                                data_root_path=args.data_root_path,
                                list_path=args.list_path,
                                split=val_split,
                                base_size=args.base_size,
                                crop_size=args.crop_size,
                                training=False,
                                class_16=args.class_16,
                                is_source=is_source)
        self.val_loader = data.DataLoader(val_set,
                                batch_size=self.args.batch_size,
                                shuffle=False,
                                num_workers=self.args.data_loader_workers,
                                pin_memory=self.args.pin_memory,
                                drop_last=True)
                                
        self.valid_iterations = (len(val_set) + self.args.batch_size) // self.args.batch_size
        self.num_iterations = (len(data_set) + self.args.batch_size) // self.args.batch_size
