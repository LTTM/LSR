import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from math import ceil
from distutils.version import LooseVersion
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt

import sys
sys.path.append(os.path.abspath('.'))
from datasets.cityscapes_Dataset import City_DataLoader, inv_preprocess, decode_labels
from datasets.crosscity_Dataset import CrossCity_DataLoader
from tools.train_source import Eval, add_train_args, str2bool, init_args
from utils.train_helper import get_model

def get_mIoU(pre_image, gt_image, num_classes):
    mask = (gt_image >= 0) & (gt_image < num_classes)
    label = num_classes * gt_image[mask].astype('int') + pre_image[mask]
    count = np.bincount(label, minlength=num_classes**2)
    confusion_matrix = count.reshape(num_classes, num_classes)
    MIoU = np.diag(confusion_matrix) / (
                np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix) )
    return 100*np.nanmean(MIoU)

class Evaluater():
    def __init__(self, args, cuda=None, train_id=None, logger=None):
        self.args = args
        self.cuda = cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda else 'cpu')

        self.current_MIoU = 0
        self.best_MIou = 0
        self.current_epoch = 0
        self.current_iter = 0
        self.train_id = train_id
        self.logger = logger

        # set TensorboardX
        self.writer = SummaryWriter(self.args.checkpoint_dir)
        if self.args.save_all_images:
            self.img_path = os.path.join(self.args.checkpoint_dir, "preds")
            os.mkdir(self.img_path)
            if self.args.save_gt:
                self.gt_path = os.path.join(self.args.checkpoint_dir, "gt")
                os.mkdir(self.gt_path)
            if self.args.save_rgb:
                self.rgb_path = os.path.join(self.args.checkpoint_dir, "rgb")
                os.mkdir(self.rgb_path)


        # Metric definition
        self.Eval = Eval(self.args.num_classes)

        # loss definition
        self.loss = nn.CrossEntropyLoss(ignore_index= -1)
        self.loss.to(self.device)

        # model
        self.model, params = get_model(self.args)
        self.model = nn.DataParallel(self.model, device_ids=[0])
        self.model.to(self.device)

        # load pretrained checkpoint
        if self.args.pretrained_ckpt_file is not None:
            path1 = os.path.join(*self.args.checkpoint_dir.split('/')[:-1], self.train_id + 'best.pth')
            path2 = self.args.pretrained_ckpt_file
            if os.path.exists(path1):
                pretrained_ckpt_file = path1
            elif os.path.exists(path2):
                pretrained_ckpt_file = path2
            else:
                raise AssertionError("no pretrained_ckpt_file")
            self.load_checkpoint(pretrained_ckpt_file)

        # dataloader
        self.dataloader = City_DataLoader(self.args) if self.args.dataset=="cityscapes" else CrossCity_DataLoader(self.args)
        if self.args.dataset=="cityscapes":
            self.dataloader.val_loader = self.dataloader.data_loader
        self.dataloader.valid_iterations = min(self.dataloader.num_iterations, 500)
        self.epoch_num = ceil(self.args.iter_max / self.dataloader.num_iterations)

    def main(self):
        # choose cuda
        if self.cuda:
            current_device = torch.cuda.current_device()
            self.logger.info("This model will run on {}".format(torch.cuda.get_device_name(current_device)))
        else:
            self.logger.info("This model will run on CPU")

        # validate
        self.validate()

        self.writer.close()

    def validate(self):
        if self.args.split == 'test':
            self.logger.info('testing one epoch...')
        else:
            self.logger.info('validating one epoch...')
        self.Eval.reset()
        with torch.no_grad():
            if self.args.split == 'test':
                tqdm_batch = tqdm(self.dataloader.val_loader, total=self.dataloader.valid_iterations, desc="Test Epoch-{}-".format(self.current_epoch + 1))
            else:
                tqdm_batch = tqdm(self.dataloader.val_loader, total=self.dataloader.valid_iterations, desc="Val Epoch-{}-".format(self.current_epoch + 1))
            self.model.eval()
            i = 0
            img_count = 0
            for x, y, y_down, id in tqdm_batch:

                i += 1
                if self.cuda:
                    x, y = x.to(self.device), y.to(device=self.device, dtype=torch.long)

                # model
                pred = self.model(x)[0]
                y = torch.squeeze(y, 1)
                label = y.cpu().numpy()

                if i==1:
                    self.logger.info('Input and Predictions have dimensions: %dx%d'%x.size()[-1:1:-1])
                    self.logger.info('Labels have dimensions: %dx%d'%y.size()[-1:0:-1])
                    self.logger.info('Features have dimensions: %dx%d'%y_down.size()[-1:0:-1])

                argpred = np.argmax(pred.data.cpu().numpy(), axis=1)

                self.Eval.add_batch(label, argpred)
                miou = get_mIoU(argpred, label, self.args.num_classes)

                if i == self.dataloader.valid_iterations:
                    break

                if self.args.save_all_images:
                    preds_colors = decode_labels(argpred, self.args.num_classes, self.args.show_num_images)[0]
                    preds_colors = preds_colors.transpose(0, 1).transpose(1,2).numpy()
                    plt.imsave(os.path.join(self.img_path, "%d_miou%.2f.png"%(id, miou)), preds_colors)
                    if self.args.save_gt:
                        label_colors = decode_labels(label, self.args.num_classes, self.args.show_num_images)[0]
                        label_colors = label_colors.transpose(0, 1).transpose(1,2).numpy()
                        plt.imsave(os.path.join(self.gt_path, "%d.png"%id), label_colors)
                    if self.args.save_rgb:
                        rgb = inv_preprocess(x.clone().cpu(), self.args.show_num_images, numpy_transform=self.args.numpy_transform)[0]
                        rgb = rgb.transpose(0, 1).transpose(1,2).numpy()
                        plt.imsave(os.path.join(self.rgb_path, "%d.png"%id), rgb)
                else:
                    if i % 20 == 0 and self.args.image_summary:
                        images_inv = inv_preprocess(x.clone().cpu(), self.args.show_num_images, numpy_transform=self.args.numpy_transform)
                        labels_colors = decode_labels(label, self.args.num_classes, self.args.show_num_images)
                        preds_colors = decode_labels(argpred, self.args.num_classes, self.args.show_num_images)
                        for index, (img, lab, color_pred) in enumerate(zip(images_inv, labels_colors, preds_colors)):
                            self.writer.add_image(str(index)+'/Images', img, img_count)
                            self.writer.add_image(str(index)+'/Labels', lab, img_count)
                            self.writer.add_image(str(index)+'/preds', color_pred, img_count)
                        img_count += 1

            # get eval result
            if self.args.class_16:
                def val_info(Eval, name):
                    PA = Eval.Pixel_Accuracy()
                    MPA_16, MPA_13 = Eval.Mean_Pixel_Accuracy()
                    MIoU_16, MIoU_13 = Eval.Mean_Intersection_over_Union()
                    FWIoU_16, FWIoU_13 = Eval.Frequency_Weighted_Intersection_over_Union()
                    PC_16, PC_13 = Eval.Mean_Precision()
                    print("########## Eval{} ############".format(name))

                    self.logger.info('\nEpoch:{:.3f}, {} PA:{:.3f}, MPA_16:{:.3f}, MIoU_16:{:.3f}, FWIoU_16:{:.3f}, PC_16:{:.3f}'.format(self.current_epoch, name, PA, MPA_16, MIoU_16, FWIoU_16, PC_16))
                    self.logger.info('\nEpoch:{:.3f}, {} PA:{:.3f}, MPA_13:{:.3f}, MIoU_13:{:.3f}, FWIoU_13:{:.3f}, PC_13:{:.3f}'.format(self.current_epoch, name, PA, MPA_13, MIoU_13, FWIoU_13, PC_13))
                    return PA, MPA_16, MIoU_16, FWIoU_16
            else:
                def val_info(Eval, name):
                    PA = Eval.Pixel_Accuracy()
                    MPA = Eval.Mean_Pixel_Accuracy()
                    MIoU = Eval.Mean_Intersection_over_Union()
                    FWIoU = Eval.Frequency_Weighted_Intersection_over_Union()
                    PC = Eval.Mean_Precision()
                    print("########## Eval{} ############".format(name))

                    self.logger.info('\nEpoch:{:.3f}, {} PA1:{:.3f}, MPA1:{:.3f}, MIoU1:{:.3f}, FWIoU1:{:.3f}, PC:{:.3f}'.format(self.current_epoch, name, PA, MPA, MIoU, FWIoU, PC))
                    return PA, MPA, MIoU, FWIoU

            PA, MPA, MIoU, FWIoU = val_info(self.Eval, "")

            self.Eval.Print_Every_class_Eval(writer=self.logger)
            tqdm_batch.close()

        return PA, MPA, MIoU, FWIoU

    def load_checkpoint(self, filename):
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            if 'state_dict' in checkpoint:
                try:
                    self.model.load_state_dict(checkpoint['state_dict'])
                except RuntimeError as e:
                    old_dict = self.model.state_dict()
                    for k in checkpoint['state_dict']:
                        if not 'layer5' in k:
                            if 'layer6' in k:
                                old_dict[k.replace('layer6', 'clas')] = checkpoint['state_dict'][k]
                            else:
                                old_dict[k] = checkpoint['state_dict'][k]
                    self.model.load_state_dict(old_dict)
            else:
                try:
                    self.model.module.load_state_dict(checkpoint)
                except RuntimeError as e:
                    old_dict = self.model.state_dict()
                    for k in checkpoint:
                        if not 'layer5' in k:
                            if 'layer6' in k:
                                old_dict['module.'+k.replace('layer6', 'clas')] = checkpoint[k]
                            else:
                                old_dict['module.'+k] = checkpoint[k]
                    self.model.load_state_dict(old_dict)
            self.logger.info("Checkpoint loaded successfully from "+filename)
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.args.checkpoint_dir))

if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('1.0.0'), 'PyTorch>=1.0.0 is required'

    file_os_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(file_os_dir)
    os.chdir('..')

    arg_parser = argparse.ArgumentParser()
    arg_parser = add_train_args(arg_parser)
    arg_parser.add_argument('--source_dataset', default='None', type=str, help='source dataset choice')

    arg_parser.add_argument('--softmax_upsample', type=str2bool, default=True, help="wether to upsample before or after the softmax layer, if true upsampling is performed after the layer")
    arg_parser.add_argument('--image_summary', type=str2bool, default=False, help="image_summary")
    arg_parser.add_argument('--save_all_images', type=str2bool, default=False, help="save images to file")
    arg_parser.add_argument('--save_gt', type=str2bool, default=False, help="whether to also save gt labels")
    arg_parser.add_argument('--save_rgb', type=str2bool, default=False, help="whether to also save rgb images")

    args = arg_parser.parse_args()
    if args.split == "train" and not args.dataset == 'city': args.split = "val"
    if args.checkpoint_dir == "none": args.checkpoint_dir = args.pretrained_ckpt_file + "/eval"
    args, train_id, logger = init_args(args)
    args.batch_size_per_gpu = 2
    args.crop_size = args.target_crop_size
    args.base_size = args.target_base_size

    args.class_13 = args.num_classes==13
    args.class_16 = args.num_classes==16

    assert (args.source_dataset == 'synthia' and args.num_classes == 16) or (args.dataset == 'city' and args.num_classes == 13) or (args.source_dataset == 'gta5' and args.num_classes == 19) or (args.source_dataset == 'cityscapes' and args.num_classes == 19), 'dataset:{0:} - classes:{1:}'.format(args.source_dataset, args.num_classes)

    agent = Evaluater(args=args, cuda=True, train_id="train_id", logger=logger)
    agent.main()
