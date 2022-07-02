import os
import torch
torch.backends.cudnn.benchmark=True
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from distutils.version import LooseVersion
import numpy as np
import cv2 as cv
import sys
sys.path.append(os.path.abspath('.'))

from datasets.synthia_Dataset import SYNTHIA_Dataset
from datasets.gta5_Dataset import GTA5_Dataset
from datasets.crosscity_Dataset import CrossCity_DataLoader, CrossCity_Dataset
from datasets.cityscapes_Dataset import City_Dataset, inv_preprocess, decode_labels

from utils.losses import normAlignment, perpendicularity, clustering, vectorsExtractor, predHistDown, predNNDown, predPDDown
from tools.train_source import Trainer, str2bool, argparse, add_train_args, init_args, datasets_path

class UDATrainer(Trainer):

    def __init__(self, args, cuda=None, train_id="None", logger=None):
        super().__init__(args, cuda, train_id, logger)


        ### DATASETS ###
        self.logger.info('Adaptation {} -> {}'.format(self.args.source_dataset, self.args.target_dataset))

        source_data_kwargs = {'data_root_path':args.data_root_path,
                              'list_path':args.list_path,
                              'base_size':args.base_size,
                              'crop_size':args.crop_size}
        target_data_kwargs = {'data_root_path': args.target_root_path,
                              'list_path': args.target_list_path,
                              'base_size': args.target_base_size,
                              'crop_size': args.target_crop_size}
        dataloader_kwargs = {'batch_size':self.args.batch_size,
                             'num_workers':self.args.data_loader_workers,
                             'pin_memory':self.args.pin_memory,
                             'drop_last':True}
        val_dataloader_kwargs = {'batch_size':1,
                              'num_workers':self.args.data_loader_workers,
                              'pin_memory':self.args.pin_memory,
                              'drop_last':True}

        if self.args.source_dataset == 'synthia':
            source_data_kwargs['class_16'] = target_data_kwargs['class_16'] = args.class_16

        if self.args.source_dataset == 'cityscapes':
            source_data_kwargs['class_13'] = target_data_kwargs['class_13'] = args.class_13

        source_data_gen = SYNTHIA_Dataset if self.args.source_dataset == 'synthia' else GTA5_Dataset if self.args.source_dataset == 'gta5' else City_Dataset

        self.source_dataloader = data.DataLoader(source_data_gen(args, split='train', is_source=True, **source_data_kwargs), shuffle=True, **dataloader_kwargs)
        source_val_dataset = source_data_gen(args, split='val', **source_data_kwargs)
        self.source_valid_iterations = len(source_val_dataset)
        self.source_val_dataloader = data.DataLoader(source_val_dataset, shuffle=False, **val_dataloader_kwargs)

        target_dataset = CrossCity_Dataset(args, **target_data_kwargs) if self.args.dataset == 'city' else City_Dataset(args, split=self.args.target_train_split, **target_data_kwargs)
        self.target_dataloader = data.DataLoader(target_dataset, shuffle=True, **dataloader_kwargs)
        target_data_set = CrossCity_Dataset(args, split='val', **target_data_kwargs) if self.args.dataset == 'city' else City_Dataset(args, split='val', **target_data_kwargs)
        self.target_val_dataloader = data.DataLoader(target_data_set, shuffle=False, **val_dataloader_kwargs)

        self.dataloader.val_loader = self.target_val_dataloader
        self.dataloader.valid_iterations = len(target_data_set)

        self.dataloader.num_iterations = min(self.dataloader.num_iterations, ((len(target_dataset) + self.args.batch_size) // self.args.batch_size)-1)

        self.ignore_index = -1
        self.current_round = self.args.init_round
        self.round_num = self.args.round_num

        self.target_shape = self.args.target_crop_size if self.args.random_crop or self.args.base_size is None else self.args.target_base_size


        # define the features spatial resolution
        if args.backbone == 'resnet101' or args.backbone == 'resnet50':
            self.target_feat_shape = (self.target_shape[0]//8+1, self.target_shape[1]//8+1) # padding error => 1 extra pixel
        else:
            self.target_feat_shape = (self.target_shape[0]//8, self.target_shape[1]//8) # correct padding
        self.logger.info("Target features have resolution: %dx%d"%self.feat_shape)

        ### LOSSES ###
        self.use_em_loss = self.args.lambda_entropy != 0.
        if self.use_em_loss:
            self.entropy_loss = IW_MaxSquareloss(ignore_index=-1,
                                                 num_class=self.args.num_classes,
                                                 ratio=self.args.IW_ratio)
            self.entropy_loss.to(self.device)

        self.best_MIou, self.best_iter, self.current_iter, self.current_epoch = None, None, None, None

        self.epoch_num = None


    def main(self):
        # display args details
        self.logger.info("Global configuration as follows:")
        for key, val in vars(self.args).items():
            self.logger.info("{:25} {}".format(key, val))

        # choose cuda
        current_device = torch.cuda.current_device()
        self.logger.info("This model will run on {}".format(torch.cuda.get_device_name(current_device)))

        # load pretrained checkpoint
        if self.args.pretrained_ckpt_file is not None:
            if os.path.isdir(self.args.pretrained_ckpt_file):
                self.args.pretrained_ckpt_file = os.path.join(self.args.checkpoint_dir, self.train_id + 'final.pth')
            self.load_checkpoint(self.args.pretrained_ckpt_file)

        if not self.args.continue_training:
            self.best_MIou, self.best_iter, self.current_iter, self.current_epoch = 0, 0, 0, 0

        if self.args.continue_training:
            self.load_checkpoint(os.path.join(self.args.checkpoint_dir, self.train_id + 'final.pth'))

        #self.args.iter_max = self.dataloader.num_iterations * self.args.epoch_each_round * self.round_num
        self.args.iter_max = 50000
        self.logger.info('Iter max: {} \nNumber of iterations: {}'.format(self.args.iter_max, self.dataloader.num_iterations))

        if self.args.log_protos:
            self.sprotos_log = [open(os.path.join(self.args.checkpoint_dir, "sproto_%d.csv")%i, 'w') for i in range(self.args.num_classes)]
        if self.args.log_twopass:
            self.down_images_path = os.path.join(self.args.checkpoint_dir, "down_preds")
            os.mkdir(self.down_images_path)

        # train
        self.train_round()

        self.writer.close()

        if self.args.log_protos:
            for f in self.sprotos_log:
                f.close()


    def train_round(self):
        for r in range(self.current_round, self.round_num):
            torch.cuda.empty_cache()
            self.logger.info("\n############## Begin {}/{} Round! #################\n".format(self.current_round + 1, self.round_num))
            self.logger.info("epoch_each_round: {}".format(self.args.epoch_each_round))

            self.epoch_num = (self.current_round + 1) * self.args.epoch_each_round

            self.train()

            self.current_round += 1
            torch.cuda.empty_cache()


    def train_one_epoch(self):
        tqdm_epoch = tqdm(zip(self.source_dataloader, self.target_dataloader), total=self.dataloader.num_iterations,
                          desc="Train Round-{}-Epoch-{}-total-{}".format(self.current_round, self.current_epoch + 1, self.epoch_num))

        self.logger.info("Training one epoch...")
        self.Eval.reset()


        # Set the model to be in training mode (for batchnorm and dropout)
        if self.args.freeze_bn:  # default False
            self.model.eval()
            self.logger.info("freeze bacth normalization successfully!")
        else:
            self.model.train()

        ### Logging setup ###

        log_list, log_strings = [None], ['souce_mcce_loss']
        if self.use_em_loss:
            log_strings.append('target_em_loss')
            log_list.append(None)

        log_string = 'epoch{}-batch-{}:' + '={:3f}-'.join(log_strings) + '={:3f}'

        batch_idx = 0
        for batch_s, batch_t in tqdm_epoch:

            self.poly_lr_scheduler(optimizer=self.optimizer, init_lr=self.args.lr)
            self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]["lr"], self.current_iter)


            torch.cuda.empty_cache()
            #######################
            # Source forward step #
            #######################

            # train data (labeled)
            x, y, y_down, info = batch_s
            if self.cuda:
                x, y, y_down = Variable(x).to(self.device, dtype=torch.float32), Variable(y).to(device=self.device, dtype=torch.long), Variable(y_down).to(self.device, dtype=torch.long)


            ########################
            # model output ->  list of:  1) pred; 2) feat from encoder's output
            pred_source, feat_source = self.model(x)
            ########################

            f_c, ib_c, n_c = self.extractor(feat_source, y_down)
            torch.cuda.empty_cache()

            # exponential smoothing
            for i in range(self.args.num_classes):
                if self.current_iter == 0 or self.b_c[i].numel() == 0:
                    self.b_c[i] = ib_c[i].clone()
                else:
                    if ib_c[i].numel() != 0:
                        self.b_c[i] = torch.clamp(self.args.centroids_smoothing *self.b_c[i].detach() + (1. - self.args.centroids_smoothing)*ib_c[i].clone(), min=0)
                    else:
                        self.b_c[i] = self.b_c[i].detach() # allows to use smoothed versions for gradient descent

            ##################################
            # Source supervised optimization #
            ##################################
            self.optimizer.zero_grad()

            y = torch.squeeze(y, 1)
            l1 = self.loss(pred_source, y)  # cross-entropy loss from train_source.py

            if self.args.norm_type == "global" or self.args.norm_type == "percent" or self.args.norm_type == "multiplicative":
                target_norm = feat_source.detach().norm(dim=1).mean() # global norm
                l2 = self.norm(feat_source, target_norm) #       version
            elif self.args.norm_type == "class":
                l2 = self.norm(f_c, n_c) # per-class norm version
            else:
                if self.num_iterations > 0:
                    l2 = self.norm_s(feat_source, norm_plane)
                    norm_plane = feat_source.detach().norm(dim=1)
                else:
                    l2 = torch.tensor([0.], requires_grad=True, device=self.device)
                    norm_plane = feat_source.detach().norm(dim=1)

            torch.cuda.empty_cache()
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
                try:
                    perp_loss.backward(retain_graph=self.use_clustering_loss)
                except RuntimeError:
                    print("Brokem protos", info)
                    continue
                cur_loss += perp_loss.item()
            if self.use_clustering_loss:
                clust_loss = self.args.lambda_cluster*l4
                clust_loss.backward()
                cur_loss += clust_loss.item()

            # log
            log_ind = 0
            log_list[log_ind] = l1.item()
            log_ind += 1

            self.writer.add_scalar('source_global_loss', cur_loss, self.current_iter)
            self.writer.add_scalar('source_norm_loss', l2.item(), self.current_iter)
            self.writer.add_scalar('source_perp_loss', l3.item(), self.current_iter)
            self.writer.add_scalar('mean_interclass_angle', 180.*np.arccos(l3.item())/np.pi, self.current_iter)
            self.writer.add_scalar('source_clust_loss', l4.item(), self.current_iter)

            if batch_idx % 50 == 0:
                avg_norms = [n.detach().mean().cpu() for n in n_c if n.numel()>0]
                self.writer.add_histogram('source_norms', np.array(avg_norms), self.current_iter)

            if self.args.norm_type == "global" or self.args.norm_type == "percent" or self.args.norm_type == "multiplicative":
                self.writer.add_scalar('norm_target', target_norm, self.current_iter)

            del x, y, y_down, pred_source, feat_source, f_c
            del l1, l2, l3, l4
            torch.cuda.empty_cache()
            #######################
            # Target forward step #
            #######################

            # target data (unlabeld)
            x, _, _, _= batch_t
            if self.cuda:
                x = Variable(x).to(self.device, dtype=torch.float32)

            ########################
            # model output ->  list of:  1) pred; 2) feat from encoder's output
            pred_target, feat_target = self.model(x)  # creates the graph
            ########################


            #####################
            # Adaptation Losses #
            #####################
            cur_loss = 0

            if self.use_em_loss:
                em_loss = self.args.lambda_entropy * self.entropy_loss(pred_target, F.softmax(pred_target, dim=1))
                em_loss.backward(retain_graph=self.use_norm_loss or self.use_orthogonality_loss or self.use_clustering_loss)

                log_list[log_ind] = em_loss.item()
                cur_loss += log_list[log_ind]
                log_ind += 1

            if self.args.target_down == 'same':
                if self.args.down_type == "hist":
                    y_down_t = predHistDown(pred_target, self.target_feat_shape, self.args.hist_th, self.args.num_classes, self.args.conf_th)
                else:
                    y_down_t = predNNDown(pred_target, self.target_feat_shape, self.args.conf_th)
            elif self.args.target_down == 'proto':
                y_down_t = predPDDown(feat_target.detach(), self.b_c, double_pass=False, conf_th=self.args.conf_th)
            else:
                if self.args.log_twopass and batch_idx % self.args.logging_interval == 0:
                    y_down_t, (old, old_filter, new) = predPDDown(feat_target.detach(), self.b_c, double_pass=True, conf_th=self.args.conf_th, log_twopass=True)
                    x1_col = inv_preprocess(x.clone().cpu(), self.args.show_num_images, numpy_transform=self.args.numpy_transform)[0].transpose(0,1).transpose(1,2).numpy()[...,::-1]
                    y1_col = decode_labels(y_down_t, self.args.num_classes, self.args.show_num_images)[0].transpose(0,1).transpose(1,2).numpy()[...,::-1]
                    y2_col = decode_labels(old, self.args.num_classes, self.args.show_num_images)[0].transpose(0,1).transpose(1,2).numpy()[...,::-1]
                    y3_col = decode_labels(old_filter, self.args.num_classes, self.args.show_num_images)[0].transpose(0,1).transpose(1,2).numpy()[...,::-1]
                    y4_col = decode_labels(new, self.args.num_classes, self.args.show_num_images)[0].transpose(0,1).transpose(1,2).numpy()[...,::-1]
                    cv.imwrite(os.path.join(self.down_images_path, "%d_rgb.png"%self.current_iter), cv.resize(np.uint8(x1_col*255.), self.target_feat_shape, interpolation=cv.INTER_AREA))
                    cv.imwrite(os.path.join(self.down_images_path, "%d_new_filter.png"%self.current_iter), np.uint8(y1_col*255.))
                    cv.imwrite(os.path.join(self.down_images_path, "%d_old.png"%self.current_iter), np.uint8(y2_col*255.))
                    cv.imwrite(os.path.join(self.down_images_path, "%d_old_filter.png"%self.current_iter), np.uint8(y3_col*255.))
                    cv.imwrite(os.path.join(self.down_images_path, "%d_new.png"%self.current_iter), np.uint8(y4_col*255.))
                    del old, old_filter, new, y1_col, y2_col, y3_col, y4_col, x1_col
                else:
                    y_down_t = predPDDown(feat_target.detach(), self.b_c, double_pass=True, conf_th=self.args.conf_th, log_twopass=False)

            f_ct, b_ct, n_ct = self.extractor(feat_target, y_down_t)
            torch.cuda.empty_cache()

            if self.args.norm_type == "global" or self.args.norm_type == "percent" or self.args.norm_type == "multiplicative":
                l2_t = self.norm(feat_target, target_norm)
            elif self.args.norm_type == "class":
                l2_t = self.norm(f_ct, n_c)
            else:
                if self.num_iterations > 0:
                    l2_t = self.norm(feat_target, norm_plane)
                    norm_plane_t = feat_target.detach().norm(dim=1)
                else:
                    l2_t = torch.tensor([0.], requires_grad=True, device=self.device)
                    norm_plane_t = feat_target.detach().norm(dim=1)


            l4_t = self.clust(f_ct, self.b_c)

            if self.use_norm_loss:
                norm_loss = self.args.lambda_norm*l2_t
                norm_loss.backward(retain_graph=self.use_clustering_loss)
                cur_loss += norm_loss.item()
            if self.use_clustering_loss:
                clust_loss = self.args.lambda_cluster*l4_t
                clust_loss.backward()
                cur_loss += clust_loss.item()
            torch.cuda.empty_cache()


            self.writer.add_scalar('target_global_loss', cur_loss, self.current_iter)
            self.writer.add_scalar('target_norm_loss', l2_t.item(), self.current_iter)
            self.writer.add_scalar('target_clust_loss', l4_t.item(), self.current_iter)

            if batch_idx % 50 == 0:
                avg_norms = [n.detach().mean().cpu() for n in n_ct if n.numel()>0]
                if len(avg_norms) > 0:
                    self.writer.add_histogram('target_norms', np.array(avg_norms), self.current_iter)


            del x, pred_target, feat_target, y_down_t, f_ct, ib_c, b_ct, n_c, n_ct
            del l2_t, l4_t
            if self.use_em_loss:
                del em_loss
            torch.cuda.empty_cache()
            #self.writer.flush()

            #break

            self.optimizer.step()

            # logging
            if batch_idx % self.args.logging_interval == 0:
                self.logger.info(log_string.format(self.current_epoch, batch_idx, *log_list))
                if self.args.log_protos:
                    for f,p in zip(self.sprotos_log, self.b_c):
                        if p.numel() > 0:
                            for n in p[0, :-1]:
                                f.write("%.3f,"%n)
                            f.write("%.3f\n"%p[0, -1])
                        else:
                            f.write("0,"*2047+"0\n")
                        f.flush()

            for name, elem in zip(log_strings, log_list):
                self.writer.add_scalar(name, elem, self.current_iter)

            if batch_idx==self.dataloader.num_iterations:
                break

            batch_idx += 1

            self.current_iter += 1

        tqdm_epoch.close()
        torch.cuda.empty_cache()

        # eval on source domain
        self.validate_source()
        torch.cuda.empty_cache()

        if self.args.save_inter_model:
            self.logger.info("Saving model of epoch {} ...".format(self.current_epoch))
            self.save_checkpoint(self.train_id + '_epoch{}.pth'.format(self.current_epoch))



def add_UDA_train_args(arg_parser):

    # shared
    arg_parser.add_argument('--source_dataset', default='gta5', type=str, choices=['gta5', 'synthia', 'cityscapes'], help='source dataset choice')
    arg_parser.add_argument('--source_split', default='train', type=str, help='source datasets split')
    arg_parser.add_argument('--init_round', type=int, default=0, help='init_round')
    arg_parser.add_argument('--round_num', type=int, default=1, help="num round")
    arg_parser.add_argument('--epoch_each_round', type=int, default=2, help="epoch each round")

    arg_parser.add_argument('--logging_interval', type=int, default=250, help="interval in steps for logging")
    arg_parser.add_argument('--save_inter_model', type=str2bool, default=False, help="save model at the end of each epoch or not")

    arg_parser.add_argument('--target_train_split', type=str, default="train", help="list file to use for training")

    # downsampling
    arg_parser.add_argument('--target_down', type=str, default='doubleproto', choices = ['same', 'proto', 'doubleproto'], help="type of downsampling to use for target feature-level maps")

    # update default values
    arg_parser.add_argument('--lambda_norm', default=0.025, type=float, help="lambda of norm loss")
    arg_parser.add_argument('--lambda_cluster', default=0.1, type=float, help="lambda of clustering loss")
    arg_parser.add_argument('--lambda_ortho', default=0.2, type=float, help="lambda of orthogonality loss")

    # entropy loss
    arg_parser.add_argument('--lambda_entropy', type=float, default=0., help="lambda of target loss")
    arg_parser.add_argument('--IW_ratio', type=float, default=0.2, help='the ratio of image-wise weighting factor')

    arg_parser.add_argument('--log_protos', type=str2bool, default=False, help="whether to log prototypes")
    arg_parser.add_argument('--log_twopass', type=str2bool, default=False, help="whether to log twopass images")



    return arg_parser



def init_UDA_args(args):

    def str2none(l):
        l = [l] if not isinstance(l,list) else l
        for i,el in enumerate(l):
            if el == 'None':
                l[i]=None
        return l if len(l)>1 else l[0]

    def str2float(l):
        for i,el in enumerate(l):
            try: l[i] = float(el)
            except (ValueError,TypeError): l[i] = el
        return l


    return args



if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('1.0.0'), 'PyTorch>=1.0.0 is required'

    file_os_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(file_os_dir)
    os.chdir('..')

    arg_parser = argparse.ArgumentParser()
    arg_parser = add_train_args(arg_parser)
    arg_parser = add_UDA_train_args(arg_parser)

    args = arg_parser.parse_args()
    args, train_id, logger = init_args(args)
    args = init_UDA_args(args)

    args.target_dataset = args.dataset

    train_id = str(args.source_dataset)+"2"+str(args.target_dataset)

    assert (args.source_dataset == 'synthia' and args.num_classes == 16) or (args.source_dataset == 'gta5' and args.num_classes == 19) or (args.source_dataset == 'cityscapes' and args.num_classes == 13), 'dataset:{0:} - classes:{1:}'.format(args.source_dataset, args.num_classes)

    agent = UDATrainer(args=args, cuda=True, train_id=train_id, logger=logger)
    agent.main()
