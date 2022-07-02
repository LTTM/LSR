import os
import random
import logging
import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
from math import ceil
import numpy as np
from distutils.version import LooseVersion
from tensorboardX import SummaryWriter

import sys
sys.path.append(os.path.abspath('.'))
from utils.eval import Eval
from utils.train_helper import get_model

from datasets.cityscapes_Dataset import City_Dataset, City_DataLoader, inv_preprocess, decode_labels
from datasets.gta5_Dataset import GTA5_DataLoader
from datasets.synthia_Dataset import SYNTHIA_DataLoader

from utils.losses import normAlignment, perpendicularity, clustering, vectorsExtractor, predHistDown, predNNDown

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

class Trainer():
    def __init__(self, args, cuda=None, train_id="None", logger=None):
        self.args = args
        self.cuda = cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda else 'cpu')
        self.train_id = train_id
        self.logger = logger

        self.current_MIoU = 0
        self.best_MIou = 0
        self.best_source_MIou = 0
        self.current_epoch = 0
        self.current_iter = 0
        self.second_best_MIou = 0

        # set TensorboardX
        self.writer = SummaryWriter(self.args.checkpoint_dir)

        # Metric definition
        self.Eval = Eval(self.args.num_classes)

        # loss definition
        self.loss = nn.CrossEntropyLoss(weight=None, ignore_index= -1)
        #self.loss = BalancedCrossEntropy()
        self.loss.to(self.device)

        # feature vectors extractor
        self.extractor = vectorsExtractor(self.args.num_classes)
        self.extractor.to(self.device)

        # extract the image size
        self.shape = self.args.crop_size if self.args.random_crop or self.args.base_size is None else self.args.base_size

        # define the features spatial resolution
        if args.backbone == 'resnet101' or args.backbone == 'resnet50':
            self.feat_shape = (self.shape[0]//8+1, self.shape[1]//8+1) # padding error => 1 extra pixel
        else:
            self.feat_shape = (self.shape[0]//8, self.shape[1]//8) # correct padding

        # norm loss
        self.use_norm_loss = self.args.lambda_norm != 0.
        if self.use_norm_loss:
            self.logger.info("Norm Loss is enabled with delta %.2E and lambda %.2E"%(self.args.delta_norm, self.args.lambda_norm))
        self.use_clustering_loss = args.lambda_cluster != 0.
        if self.use_clustering_loss:
            self.logger.info("clustering Loss is enabled with lambda %.2E"%(self.args.lambda_cluster))
        self.use_orthogonality_loss = args.lambda_ortho != 0. #<- perp
        if self.use_orthogonality_loss:
            self.logger.info("Orthogonality Loss is enabled with lambda %.2E"%(self.args.lambda_ortho))

        if self.use_clustering_loss or self.use_orthogonality_loss:
            if self.args.down_type == 'hist':
                self.logger.info("Histogram Downsampling threshold: %.2E, Confidence threshold: %.2E"%(self.args.hist_th, self.args.conf_th))
            else:
                self.logger.info("Nearest Neighbour Downsampling. Confidence threshold: %.2E"%(self.args.conf_th))

        self.logger.info("Features have resolution: %dx%d"%self.feat_shape)

        # norm alignment
        self.norm = normAlignment(delta_norm=self.args.delta_norm, ntype=self.args.norm_type, filter_norms=self.args.filter_norms)
        self.norm.to(self.device)

        # perpendicularity
        self.perp = perpendicularity()
        self.perp.to(self.device)

        # clustering
        self.clust = clustering(type=self.args.clust_type)
        self.clust.to(self.device)

        # model
        self.model, params = get_model(self.args)
        self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

        # define the features spatial resolution
        if args.backbone == 'resnet101' or args.backbone == 'resnet50':
            self.feature_channels = 2048
        else:
            self.feature_channels = 1024
        self.logger.info("Features have %d channels"%self.feature_channels)

        self.optimizer = torch.optim.SGD(
            params=params,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay
        )

        # dataloader
        if self.args.dataset=="cityscapes":
            self.dataloader = City_DataLoader(self.args, is_source=True)
        elif self.args.dataset=="gta5":
            self.dataloader = GTA5_DataLoader(self.args, is_source=True)
        else:
            self.dataloader = SYNTHIA_DataLoader(self.args, is_source=True)

        if self.args.use_target_val:
            target_data_set = City_Dataset(args,
                                           data_root_path=self.args.target_root_path,
                                           list_path=self.args.target_list_path,
                                           split='test',
                                           base_size=args.target_base_size,
                                           crop_size=args.target_crop_size,
                                           class_16=args.class_16,
                                           class_13=args.class_13)
            self.target_val_dataloader = data.DataLoader(target_data_set,
                                                         batch_size=1,
                                                         shuffle=False,
                                                         num_workers=self.args.data_loader_workers,
                                                         pin_memory=self.args.pin_memory,
                                                         drop_last=True)
            self.dataloader.valid_iterations = len(target_data_set)
            self.dataloader.val_loader = self.target_val_dataloader

        self.dataloader.num_iterations = min(self.dataloader.num_iterations, self.args.iter_max_epoch)
        self.epoch_num = ceil(self.args.iter_max / self.dataloader.num_iterations) if self.args.iter_stop is None else \
                            ceil(self.args.iter_stop / self.dataloader.num_iterations)

        # prototypes
        self.b_c = []
        for i in range(self.args.num_classes):
            self.b_c.append(torch.zeros((0,self.feature_channels), device=self.device))

    def main(self):
        # display args details
        self.logger.info("Global configuration as follows:")
        for key, val in vars(self.args).items():
            self.logger.info("{:16} {}".format(key, val))

        # choose cuda
        if self.cuda:
            current_device = torch.cuda.current_device()
            self.logger.info("This model will run on {}".format(torch.cuda.get_device_name(current_device)))
        else:
            self.logger.info("This model will run on CPU")

        # load pretrained checkpoint
        if self.args.pretrained_ckpt_file is not None:
            if os.path.isdir(self.args.pretrained_ckpt_file):
                self.args.pretrained_ckpt_file = os.path.join(self.args.checkpoint_dir, self.train_id + 'best.pth')
            self.load_checkpoint(self.args.pretrained_ckpt_file)

        if self.args.continue_training:
            self.load_checkpoint(os.path.join(self.args.checkpoint_dir, self.train_id + 'best.pth'))
            self.best_iter = self.current_iter
            self.best_source_iter = self.current_iter
        else:
            self.current_epoch = 0
        # train

        self.train()

        self.writer.close()

    def train(self):
        torch.cuda.empty_cache()
        for epoch in tqdm(range(self.current_epoch, self.epoch_num),
                          desc="Total {} epochs".format(self.epoch_num)):
            self.train_one_epoch()
            torch.cuda.empty_cache()

            self.current_epoch += 1 # to avoid overwritting first validations


            # validate
            PA, MPA, MIoU, FWIoU = self.validate()
            self.writer.add_scalar('PA', PA, self.current_epoch)
            self.writer.add_scalar('MPA', MPA, self.current_epoch)
            self.writer.add_scalar('MIoU', MIoU, self.current_epoch)
            self.writer.add_scalar('FWIoU', FWIoU, self.current_epoch)

            self.current_MIoU = MIoU
            is_best = MIoU > self.best_MIou
            if is_best:
                self.best_MIou = MIoU
                self.best_iter = self.current_iter
                self.logger.info("=>saving a new best checkpoint...")
                self.save_checkpoint(self.train_id+'best.pth')
            else:
                self.logger.info("=> The MIoU of val does't improve.")
                self.logger.info("=> The best MIoU of val is {} at {}".format(self.best_MIou, self.best_iter))

            # self.current_epoch += 1

        state = {
            'epoch': self.current_epoch + 1,
            'iteration': self.current_iter,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_MIou': self.current_MIoU
        }
        self.logger.info("=>best_MIou {} at {}".format(self.best_MIou, self.best_iter))
        self.logger.info("=>saving the final checkpoint to " + os.path.join(self.args.checkpoint_dir, self.train_id+'final.pth'))
        self.save_checkpoint(self.train_id+'final.pth')
        torch.cuda.empty_cache()

    def train_one_epoch(self):
        tqdm_epoch = tqdm(self.dataloader.data_loader, total=self.dataloader.num_iterations,
                          desc="Train Epoch-{}-total-{}".format(self.current_epoch+1, self.epoch_num))
        self.logger.info("Training one epoch...")
        self.Eval.reset()

        train_loss = []
        loss_seg_value_2 = 0
        iter_num = self.dataloader.num_iterations

        if self.args.freeze_bn:
            self.model.eval()
            self.logger.info("freeze bacth normalization successfully!")
        else:
            self.model.train()
        # Initialize your average meters

        batch_idx = 0
        for x, y, gt_down, _ in tqdm_epoch:
            self.poly_lr_scheduler(
                optimizer=self.optimizer,
                init_lr=self.args.lr,
                iter=self.current_iter,
                max_iter=self.args.iter_max,
                power=self.args.poly_power,
            )
            if self.args.iter_stop is not None and self.current_iter >= self.args.iter_stop:
                self.logger.info("iteration arrive {}(early stop)/{}(total step)!".format(self.args.iter_stop, self.args.iter_max))
                break
            if self.current_iter >= self.args.iter_max:
                self.logger.info("iteration arrive {}!".format(self.args.iter_max))
                break

            if self.cuda:
                x, y, y_down = x.to(self.device, dtype=torch.float32), y.to(device=self.device, dtype=torch.long), gt_down.to(self.device, dtype=torch.long)

            y = torch.squeeze(y, 1)
            self.optimizer.zero_grad()

            # model
            pred, feats = self.model(x)
            f_c, ib_c, n_c = self.extractor(feats, y_down)

            # exponential smoothing
            for i in range(self.args.num_classes):
                if self.current_iter == 0 or self.b_c[i].numel() == 0:
                    self.b_c[i] = ib_c[i].clone()
                else:
                    if ib_c[i].numel() != 0:
                        self.b_c[i] = torch.clamp(self.args.centroids_smoothing *self.b_c[i].detach() + (1. - self.args.centroids_smoothing)*ib_c[i].clone(), min=0)
                    else:
                        self.b_c[i] = self.b_c[i].detach() # allows to use smoothed versions for gradient descent

            # loss
            l1 = self.loss(pred, y)


            if self.args.norm_type == "global" or self.args.norm_type == "percent" or self.args.norm_type == "multiplicative":
                target_norm = feats.detach().norm(dim=1).mean() # global norm
                l2 = self.norm(feats, target_norm) #       version
            elif self.args.norm_type == "class":
                l2 = self.norm(f_c, n_c) # per-class norm version
            else:
                if self.num_iterations > 0:
                    l2 = self.norm(feats, norm_plane)
                    norm_plane = feats.detach().norm(dim=1)
                else:
                    l2 = torch.tensor([0.], requires_grad=True, device=self.device)
                    norm_plane = feats.detach().norm(dim=1)


            l3 = self.perp(self.b_c)
            l4 = self.clust(f_c, self.b_c)

            # backpropagate the losses gradients
            # separately to reduce the impact of the secondary losses
            # on the optimization of the cross categorical entropy
            l1.backward(retain_graph=self.use_norm_loss or self.use_orthogonality_loss or self.use_clustering_loss)
            cur_loss = l1.item()
            if self.use_norm_loss:
                norm_loss = self.args.lambda_norm*l2
                norm_loss.backward(retain_graph=self.use_orthogonality_loss or self.use_clustering_loss)
                cur_loss += norm_loss.item()
            if self.use_orthogonality_loss:
                perp_loss = self.args.lambda_ortho*l3
                perp_loss.backward(retain_graph=self.use_clustering_loss)
                cur_loss += perp_loss.item()
            if self.use_clustering_loss:
                clust_loss = self.args.lambda_cluster*l4
                clust_loss.backward()
                cur_loss += clust_loss.item()

            # optimizer
            self.optimizer.step()

            train_loss.append(cur_loss)


            self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]["lr"], self.current_iter)
            self.writer.add_scalar('loss_global', cur_loss, self.current_iter)
            self.writer.add_scalar('loss_mcce', l1.item(), self.current_iter)
            self.writer.add_scalar('loss_norm', l2.item(), self.current_iter)
            self.writer.add_scalar('loss_perp', l3.item(), self.current_iter)
            self.writer.add_scalar('mean_interclass_angle', 180.*np.arccos(l3.item())/np.pi, self.current_iter)
            self.writer.add_scalar('loss_clust', l4.item(), self.current_iter)

            if batch_idx % 50 == 0:
                avg_norms = [n.detach().mean().cpu() for n in n_c if n.numel()>0]
                self.writer.add_histogram('source_norms', np.array(avg_norms), self.current_iter)

            if self.args.norm_type == "global" or self.args.norm_type == "percent" or self.args.norm_type == "multiplicative":
                self.writer.add_scalar('norm_target', target_norm, self.current_iter)

            if batch_idx % 1000 == 0:
                self.logger.info("The train loss of epoch{}-batch-{}:{}".format(self.current_epoch, batch_idx, cur_loss))

            batch_idx += 1

            self.current_iter += 1

            if np.isnan(float(cur_loss)):
                raise ValueError('Loss is nan during training...')

            pred = pred.data.cpu().numpy()
            label = y.cpu().numpy()
            argpred = np.argmax(pred, axis=1)
            self.Eval.add_batch(label, argpred)

            if batch_idx==self.dataloader.num_iterations:
                break

        self.log_one_train_epoch(x, label, argpred, train_loss)
        tqdm_epoch.close()

    def log_one_train_epoch(self, x, label, argpred, train_loss):
        #show train image on tensorboard
        images_inv = inv_preprocess(x.clone().cpu(), self.args.show_num_images, numpy_transform=self.args.numpy_transform)
        labels_colors = decode_labels(label, self.args.num_classes, self.args.show_num_images)
        preds_colors = decode_labels(argpred, self.args.num_classes, self.args.show_num_images)
        for index, (img, lab, color_pred) in enumerate(zip(images_inv, labels_colors, preds_colors)):
            self.writer.add_image('train/'+ str(index)+'/Images', img, self.current_epoch)
            self.writer.add_image('train/'+ str(index)+'/Labels', lab, self.current_epoch)
            self.writer.add_image('train/'+ str(index)+'/preds', color_pred, self.current_epoch)

        if self.args.class_16:
            PA = self.Eval.Pixel_Accuracy()
            MPA_16, MPA = self.Eval.Mean_Pixel_Accuracy()
            MIoU_16, MIoU = self.Eval.Mean_Intersection_over_Union()
            FWIoU_16, FWIoU = self.Eval.Frequency_Weighted_Intersection_over_Union()
        else:
            PA = self.Eval.Pixel_Accuracy()
            MPA = self.Eval.Mean_Pixel_Accuracy()
            MIoU = self.Eval.Mean_Intersection_over_Union()
            FWIoU = self.Eval.Frequency_Weighted_Intersection_over_Union()

        self.logger.info('\nEpoch:{}, train PA1:{}, MPA1:{}, MIoU1:{}, FWIoU1:{}'.format(self.current_epoch, PA, MPA,
                                                                                       MIoU, FWIoU))
        self.writer.add_scalar('train_PA', PA, self.current_epoch)
        self.writer.add_scalar('train_MPA', MPA, self.current_epoch)
        self.writer.add_scalar('train_MIoU', MIoU, self.current_epoch)
        self.writer.add_scalar('train_FWIoU', FWIoU, self.current_epoch)

        tr_loss = sum(train_loss)/len(train_loss) if isinstance(train_loss, list) else train_loss
        self.writer.add_scalar('train_loss', tr_loss, self.current_epoch)
        tqdm.write("The average loss of train epoch-{}-:{}".format(self.current_epoch, tr_loss))

    def validate(self, mode='val'):
        self.logger.info('\nvalidating one epoch...')
        self.Eval.reset()
        with torch.no_grad():
            tqdm_batch = tqdm(self.dataloader.val_loader, total=self.dataloader.valid_iterations,
                              desc="Val Epoch-{}-".format(self.current_epoch))
            if mode == 'val':
                self.model.eval()

            i = 0

            for x, y, y_down, id in tqdm_batch:
                if self.cuda:
                    x, y, y_down = x.to(self.device, dtype=torch.float32), y.to(device=self.device, dtype=torch.long), y_down.to(self.device, dtype=torch.long)

                # model
                pred = self.model(x)[0]
                #pred_down = predHistDown(pred, y_down.shape[-2:][::-1], self.args.hist_th, self.args.num_classes, self.args.conf_th)
                if self.args.down_type == "hist":
                    pred_down = predHistDown(pred, y_down.shape[-2:][::-1], self.args.hist_th, self.args.num_classes, self.args.conf_th)
                else:
                    pred_down = predNNDown(pred, y_down.shape[-2:][::-1], self.args.conf_th)
                y = torch.squeeze(y, 1)


                pred = pred.data.cpu().numpy()
                pred_down = pred_down.data.cpu().numpy()
                label = y.cpu().numpy()
                argpred = np.argmax(pred, axis=1)

                self.Eval.add_batch(label, argpred)


            #show val result on tensorboard
            images_inv = inv_preprocess(x.clone().cpu(), self.args.show_num_images, numpy_transform=self.args.numpy_transform)
            labels_colors = decode_labels(label, self.args.num_classes, self.args.show_num_images)
            labels_down_colors = decode_labels(y_down, self.args.num_classes, self.args.show_num_images)
            preds_colors = decode_labels(argpred, self.args.num_classes, self.args.show_num_images)
            preds_down_colors = decode_labels(pred_down, self.args.num_classes, self.args.show_num_images)
            for index, (img, lab, lab_down, color_pred, color_down_pred) in enumerate(zip(images_inv, labels_colors, labels_down_colors, preds_colors, preds_down_colors)):
                self.writer.add_image(str(index)+'/Images', img, self.current_epoch)
                self.writer.add_image(str(index)+'/Labels', lab, self.current_epoch)
                self.writer.add_image(str(index)+'/Labels_down', lab_down, self.current_epoch)
                self.writer.add_image(str(index)+'/preds', color_pred, self.current_epoch)
                self.writer.add_image(str(index)+'/pseudoLabels_down', color_down_pred, self.current_epoch)

            if self.args.class_16:
                def val_info(Eval, name):
                    PA = Eval.Pixel_Accuracy()
                    MPA_16, MPA_13 = Eval.Mean_Pixel_Accuracy()
                    MIoU_16, MIoU_13 = Eval.Mean_Intersection_over_Union()
                    FWIoU_16, FWIoU_13 = Eval.Frequency_Weighted_Intersection_over_Union()
                    PC_16, PC_13 = Eval.Mean_Precision()
                    print("########## Eval{} ############".format(name))

                    self.logger.info('\nEpoch:{:.3f}, {} PA:{:.3f}, MPA_16:{:.3f}, MIoU_16:{:.3f}, FWIoU_16:{:.3f}, PC_16:{:.3f}'.format(self.current_epoch, name, PA, MPA_16,
                                                                                                MIoU_16, FWIoU_16, PC_16))
                    self.logger.info('\nEpoch:{:.3f}, {} PA:{:.3f}, MPA_13:{:.3f}, MIoU_13:{:.3f}, FWIoU_13:{:.3f}, PC_13:{:.3f}'.format(self.current_epoch, name, PA, MPA_13,
                                                                                                MIoU_13, FWIoU_13, PC_13))
                    self.writer.add_scalar('PA'+name, PA, self.current_epoch)
                    self.writer.add_scalar('MPA_16'+name, MPA_16, self.current_epoch)
                    self.writer.add_scalar('MIoU_16'+name, MIoU_16, self.current_epoch)
                    self.writer.add_scalar('FWIoU_16'+name, FWIoU_16, self.current_epoch)
                    self.writer.add_scalar('MPA_13'+name, MPA_13, self.current_epoch)
                    self.writer.add_scalar('MIoU_13'+name, MIoU_13, self.current_epoch)
                    self.writer.add_scalar('FWIoU_13'+name, FWIoU_13, self.current_epoch)
                    return PA, MPA_13, MIoU_13, FWIoU_13
            else:
                def val_info(Eval, name):
                    PA = Eval.Pixel_Accuracy()
                    MPA = Eval.Mean_Pixel_Accuracy()
                    MIoU = Eval.Mean_Intersection_over_Union()
                    FWIoU = Eval.Frequency_Weighted_Intersection_over_Union()
                    PC = Eval.Mean_Precision()
                    print("########## Eval{} ############".format(name))

                    self.logger.info('\nEpoch:{:.3f}, {} PA1:{:.3f}, MPA1:{:.3f}, MIoU1:{:.3f}, FWIoU1:{:.3f}, PC:{:.3f}'.format(self.current_epoch, name, PA, MPA,
                                                                                                MIoU, FWIoU, PC))
                    self.writer.add_scalar('PA'+name, PA, self.current_epoch)
                    self.writer.add_scalar('MPA'+name, MPA, self.current_epoch)
                    self.writer.add_scalar('MIoU'+name, MIoU, self.current_epoch)
                    self.writer.add_scalar('FWIoU'+name, FWIoU, self.current_epoch)
                    return PA, MPA, MIoU, FWIoU

            PA, MPA, MIoU, FWIoU = val_info(self.Eval, "")
            tqdm_batch.close()

        return PA, MPA, MIoU, FWIoU

    def validate_source(self):
        self.logger.info('\nvalidating source domain...')
        self.Eval.reset()
        with torch.no_grad():
            tqdm_batch = tqdm(self.source_val_dataloader, total=self.source_valid_iterations,
                              desc="Source Val Epoch-{}-".format(self.current_epoch + 1))

            self.model.eval()
            i = 0
            for x, y, y_down, id in tqdm_batch:
                if self.cuda:
                    x, y = x.to(self.device), y.to(device=self.device, dtype=torch.long)

                # model output ->  list of:  1) pred; 2) feat from encoder's output
                pred = self.model(x)[0]
                if self.args.down_type == "hist":
                    pred_down = predHistDown(pred, y_down.shape[-2:][::-1], self.args.hist_th, self.args.num_classes, self.args.conf_th)
                else:
                    pred_down = predNNDown(pred, y_down.shape[-2:][::-1], self.args.conf_th)
                y = torch.squeeze(y, 1)

                pred = pred.data.cpu().numpy()
                label = y.cpu().numpy()
                argpred = np.argmax(pred, axis=1)

                self.Eval.add_batch(label, argpred)

                #break

                i += 1
                if i == self.dataloader.valid_iterations:
                    break

            print(y_down.shape, y_down.dtype)
            #show val result on tensorboard
            images_inv = inv_preprocess(x.clone().cpu(), self.args.show_num_images, numpy_transform=self.args.numpy_transform)
            labels_colors = decode_labels(label, self.args.num_classes, self.args.show_num_images)
            labels_down_colors = decode_labels(y_down, self.args.num_classes, self.args.show_num_images)
            preds_colors = decode_labels(argpred, self.args.num_classes, self.args.show_num_images)
            preds_down_colors = decode_labels(pred_down, self.args.num_classes, self.args.show_num_images)
            for index, (img, lab, lab_down, color_pred, color_down_pred) in enumerate(zip(images_inv, labels_colors, labels_down_colors, preds_colors, preds_down_colors)):
                self.writer.add_image('source_eval/'+str(index)+'/Images', img, self.current_epoch)
                self.writer.add_image('source_eval/'+str(index)+'/Labels', lab, self.current_epoch)
                self.writer.add_image('source_eval/'+str(index)+'/Labels_down', lab_down, self.current_epoch)
                self.writer.add_image('source_eval/'+str(index)+'/preds', color_pred, self.current_epoch)
                self.writer.add_image('source_eval/'+str(index)+'/pseudoLabels_down', color_down_pred, self.current_epoch)

            if self.args.class_16:
                def source_val_info(Eval, name):
                    PA = Eval.Pixel_Accuracy()
                    MPA_16, MPA_13 = Eval.Mean_Pixel_Accuracy()
                    MIoU_16, MIoU_13 = Eval.Mean_Intersection_over_Union()
                    FWIoU_16, FWIoU_13 = Eval.Frequency_Weighted_Intersection_over_Union()
                    PC_16, PC_13 = Eval.Mean_Precision()
                    print("########## Source Eval{} ############".format(name))

                    self.logger.info('\nEpoch:{:.3f}, source {} PA:{:.3f}, MPA_16:{:.3f}, MIoU_16:{:.3f}, FWIoU_16:{:.3f}, PC_16:{:.3f}'.format(self.current_epoch, name, PA, MPA_16,
                                                                                                MIoU_16, FWIoU_16, PC_16))
                    self.logger.info('\nEpoch:{:.3f}, source {} PA:{:.3f}, MPA_13:{:.3f}, MIoU_13:{:.3f}, FWIoU_13:{:.3f}, PC_13:{:.3f}'.format(self.current_epoch, name, PA, MPA_13,
                                                                                                MIoU_13, FWIoU_13, PC_13))
                    self.writer.add_scalar('source_PA'+name, PA, self.current_epoch)
                    self.writer.add_scalar('source_MPA_16'+name, MPA_16, self.current_epoch)
                    self.writer.add_scalar('source_MIoU_16'+name, MIoU_16, self.current_epoch)
                    self.writer.add_scalar('source_FWIoU_16'+name, FWIoU_16, self.current_epoch)
                    self.writer.add_scalar('source_MPA_13'+name, MPA_13, self.current_epoch)
                    self.writer.add_scalar('source_MIoU_13'+name, MIoU_13, self.current_epoch)
                    self.writer.add_scalar('source_FWIoU_13'+name, FWIoU_13, self.current_epoch)
                    return PA, MPA_13, MIoU_13, FWIoU_13
            else:
                def source_val_info(Eval, name):
                    PA = Eval.Pixel_Accuracy()
                    MPA = Eval.Mean_Pixel_Accuracy()
                    MIoU = Eval.Mean_Intersection_over_Union()
                    FWIoU = Eval.Frequency_Weighted_Intersection_over_Union()
                    PC = Eval.Mean_Precision()

                    self.writer.add_scalar('source_PA'+name, PA, self.current_epoch)
                    self.writer.add_scalar('source_MPA'+name, MPA, self.current_epoch)
                    self.writer.add_scalar('source_MIoU'+name, MIoU, self.current_epoch)
                    self.writer.add_scalar('source_FWIoU'+name, FWIoU, self.current_epoch)
                    print("########## Source Eval{} ############".format(name))

                    self.logger.info('\nEpoch:{:.3f}, source {} PA1:{:.3f}, MPA1:{:.3f}, MIoU1:{:.3f}, FWIoU1:{:.3f}, PC:{:.3f}'.format(self.current_epoch, name, PA, MPA,
                                                                                                MIoU, FWIoU, PC))
                    return PA, MPA, MIoU, FWIoU

            PA, MPA, MIoU, FWIoU = source_val_info(self.Eval, "")
            tqdm_batch.close()

        is_best = MIoU > self.best_source_MIou
        if is_best:
            self.best_source_MIou = MIoU
            self.best_source_iter = self.current_iter
            self.logger.info("=>saving a new best source checkpoint...")
            self.save_checkpoint(self.train_id+'source_best.pth')
        else:
            self.logger.info("=> The source MIoU of val does't improve.")
            self.logger.info("=> The best source MIoU of val is {} at {}".format(self.best_source_MIou, self.best_source_iter))

        return PA, MPA, MIoU, FWIoU

    def save_checkpoint(self, filename=None):
        """
        Save checkpoint if a new best is achieved
        :param state:
        :param is_best:
        :param filepath:
        :return:
        """
        filename = os.path.join(self.args.checkpoint_dir, filename)
        state = {
            'epoch': self.current_epoch + 1,
            'iteration': self.current_iter,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_MIou':self.best_MIou
        }
        torch.save(state, filename)

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
                self.model.module.load_state_dict(checkpoint)
            self.logger.info("Checkpoint loaded successfully from "+filename)
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.args.checkpoint_dir))
            self.logger.info("**First time to train**")

    def poly_lr_scheduler(self, optimizer, init_lr=None, iter=None,
                            max_iter=None, power=None):
        init_lr = self.args.lr if init_lr is None else init_lr
        iter = self.current_iter if iter is None else iter
        max_iter = self.args.iter_max if max_iter is None else max_iter
        power = self.args.poly_power if power is None else power
        new_lr = init_lr * (1 - float(iter) / max_iter) ** power
        optimizer.param_groups[0]["lr"] = new_lr
        if len(optimizer.param_groups) == 2:
            optimizer.param_groups[1]["lr"] = 10 * new_lr


def add_train_args(arg_parser):
    # Path related arguments
    arg_parser.add_argument('--data_root_path', type=str, default=None, help="the path to dataset")
    arg_parser.add_argument('--list_path', type=str, default=None, help="the path to data split lists")
    arg_parser.add_argument('--use_target_val', type=str2bool, default=True, help="whether to continue use target dataset for validation")
    arg_parser.add_argument('--target_root_path', type=str, default=None, help="the path to target dataset")
    arg_parser.add_argument('--target_list_path', type=str, default=None, help="the path to target data split lists")
    arg_parser.add_argument('--checkpoint_dir', default="./log/train", help="the path to ckpt file")

    # Model related arguments
    arg_parser.add_argument('--backbone', default='resnet101', choices=['resnet101','vgg16','resnet50','vgg13'],  help="backbone encoder")
    arg_parser.add_argument('--bn_momentum', type=float, default=0.1, help="batch normalization momentum")
    arg_parser.add_argument('--imagenet_pretrained', type=str2bool, default=True, help="whether apply imagenet pretrained weights")
    arg_parser.add_argument('--pretrained_ckpt_file', type=str, default=None, help="whether to apply pretrained checkpoint")
    arg_parser.add_argument('--continue_training', type=str2bool, default=False, help="whether to continue training ")
    arg_parser.add_argument('--show_num_images', type=int, default=2, help="show how many images during validate")

    # train related arguments
    arg_parser.add_argument('--seed', default=12345, type=int, help='random seed')
    arg_parser.add_argument('--gpu', type=str, default="0", help=" the num of gpu")
    arg_parser.add_argument('--batch_size_per_gpu', default=1, type=int, help='input batch size')

    # dataset related arguments
    arg_parser.add_argument('--dataset', default='cityscapes', type=str, help='dataset choice')
    arg_parser.add_argument('--base_size', default="1280,720", type=str,  help='crop size of image')
    arg_parser.add_argument('--crop_size', default="1280,720", type=str, help='base size of image')
    arg_parser.add_argument('--target_base_size', default="1280,640", type=str, help='crop size of target image')
    arg_parser.add_argument('--target_crop_size', default="1280,640", type=str, help='base size of target image')

    arg_parser.add_argument('--num_classes', default=19, type=int, help='num class of mask')
    arg_parser.add_argument('--data_loader_workers', default=16, type=int, help='num_workers of Dataloader')
    arg_parser.add_argument('--pin_memory', default=2, type=int, help='pin_memory of Dataloader')
    arg_parser.add_argument('--split', type=str, default='train', help="choose from train/val/test/trainval/all")
    arg_parser.add_argument('--random_mirror', default=True, type=str2bool, help='add random_mirror')
    arg_parser.add_argument('--random_crop', default=False, type=str2bool, help='add random_crop')
    arg_parser.add_argument('--resize', default=True, type=str2bool, help='resize')
    arg_parser.add_argument('--random_resize', default=False, type=str2bool, help='resize')
    arg_parser.add_argument('--gaussian_blur', default=True, type=str2bool, help='add gaussian_blur')
    arg_parser.add_argument('--numpy_transform', default=True, type=str2bool, help='image transform with numpy style')
    arg_parser.add_argument('--color_jitter', default=True, type=str2bool, help='randomly shift white balance of images')

    # optimization related arguments
    arg_parser.add_argument('--freeze_bn', type=str2bool, default=False, help="whether freeze BatchNormalization")
    arg_parser.add_argument('--optim', default="SGD", type=str, help='optimizer')
    arg_parser.add_argument('--momentum', type=float, default=0.9)
    arg_parser.add_argument('--weight_decay', type=float, default=5e-4)

    arg_parser.add_argument('--lr', type=float, default=2.5e-4,  help="init learning rate ")
    arg_parser.add_argument('--iter_max', type=int, default=250000,  help="the maxinum of iteration")
    arg_parser.add_argument('--iter_max_epoch', type=int, default=5000,  help="maximum number of iterations per epoch")
    arg_parser.add_argument('--iter_stop', type=int, default=None, help="the early stop step")
    arg_parser.add_argument('--poly_power', type=float, default=0.9, help="poly_power")

    # clustering
    arg_parser.add_argument('--down_type', type=str, default='hist', choices = ['hist', 'nearest'], help="type of downsampling to use for source feature-level maps")
    arg_parser.add_argument('--weighted_hist', type=str2bool, default=True, help="Wether to use the weighted version of histogram downsampling")
    arg_parser.add_argument('--lambda_cluster', default=0., type=float, help="lambda of clustering loss")
    arg_parser.add_argument('--clust_type', type=str, default='absolute', choices = ['absolute', 'euclidean', 'cosine', 'percent'], help="type of clustering")
    arg_parser.add_argument('--hist_th', type=float, default=0.5, help="histogram downsampling threshold")
    arg_parser.add_argument('--centroids_smoothing', type=float, default=0.8, help="centroids exponential smoothing rate")
    arg_parser.add_argument('--conf_th', type=float, default=0.5, help="prediction confidence threshold")

    # orthogonality
    arg_parser.add_argument('--lambda_ortho', default=0., type=float, help="lambda of orthogonality loss")

    # norm
    arg_parser.add_argument('--lambda_norm', default=0., type=float, help="lambda of norm loss")
    arg_parser.add_argument('--delta_norm', default=0.1, type=float, help="norm enchancement rate")
    arg_parser.add_argument('--filter_norms', default=True, type=str2bool, help="wether to stop the gradient on smaller activations")
    arg_parser.add_argument('--norm_type', type=str, default='percent', choices = ['global', 'percent', 'class', 'multiplicative'], help="type of norm")

    return arg_parser

def init_args(args):
    args.batch_size = args.batch_size_per_gpu * ceil(len(args.gpu) / 2)

    train_id = str(args.dataset)

    crop_size = args.crop_size.split(',')
    base_size = args.base_size.split(',')
    if len(crop_size)==1:
        args.crop_size = int(crop_size[0])
        args.base_size = int(base_size[0])
    else:
        args.crop_size = (int(crop_size[0]), int(crop_size[1]))
        args.base_size = (int(base_size[0]), int(base_size[1]))

    target_crop_size = args.target_crop_size.split(',')
    target_base_size = args.target_base_size.split(',')
    if len(target_crop_size)==1:
        args.target_crop_size = int(target_crop_size[0])
        args.target_base_size = int(target_base_size[0])
    else:
        args.target_crop_size = (int(target_crop_size[0]), int(target_crop_size[1]))
        args.target_base_size = (int(target_base_size[0]), int(target_base_size[1]))

    if not args.continue_training:
        if os.path.exists(args.checkpoint_dir):
            print("checkpoint dir exists, which will be removed")
            import shutil
            shutil.rmtree(args.checkpoint_dir, ignore_errors=True)
        # print(os.getcwd())
        try:
            os.mkdir(args.checkpoint_dir)
        except FileNotFoundError:
            print('Missing parent folder in path:  {}'.format(args.checkpoint_dir))
            exit()

    if args.data_root_path is None:
        args.data_root_path = datasets_path[args.dataset]['data_root_path']
        args.list_path = datasets_path[args.dataset]['list_path']

    args.class_16 = True if args.num_classes == 16 else False
    args.class_13 = True if args.num_classes == 13 else False

    # logger configure
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(args.checkpoint_dir, 'train_log.txt'))
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    #set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.backends.cudnn.benchmark=True

    return args, train_id, logger

if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('1.0.0'), 'PyTorch>=1.0.0 is required'

    file_os_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(file_os_dir)
    os.chdir('..')

    arg_parser = argparse.ArgumentParser()
    arg_parser = add_train_args(arg_parser)

    args = arg_parser.parse_args()
    args, train_id, logger = init_args(args)

    if args.dataset == 'synthia' or args.dataset == 'gta5':
        assert (args.dataset == 'synthia' and args.num_classes == 16) or (args.dataset == 'gta5' and args.num_classes == 19), 'dataset:{0:} - classes:{1:}'.format(args.dataset, args.num_classes)

    agent = Trainer(args=args, cuda=True, train_id=train_id, logger=logger)
    agent.main()
