import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datasets.histDown import histDown
from PIL import Image


class normAlignment(nn.Module):
    def __init__(self, delta_norm=0.2, ntype="global", filter_norms=True):
        super(normAlignment, self).__init__()
        self.delta_norm = delta_norm
        self.ntype = ntype
        self.filter_norms = filter_norms
        return # else it is a vector of length num_classes

    def forward(self, inputs, target):

        if self.filter_norms or (self.ntype == "global" or self.ntype == "percent" or self.ntype == "multiplicative"):
            m = torch.mean(inputs.detach(), dim=1)
            inputs = inputs.clone()
            inputs[inputs<m] = 0 # filter out disabled activations, replacing them with 0s (stopping the gradient)

        if self.ntype == "global":
            # single, global target norm
            # absolute difference
            norms = torch.norm(inputs, dim=1)
            deltas = norms - (target+self.delta_norm).expand_as(norms)
            loss = torch.mean(torch.abs(deltas))

            return loss
        elif self.ntype == "percent":
            # single, global target norm
            # percent difference
            norms = torch.norm(inputs, dim=1)
            deltas = norms - (target+self.delta_norm).expand_as(norms)
            deltas /= target
            loss = torch.mean(torch.abs(deltas))

            return loss
        elif self.ntype == "class":
            # class-wise norms, absolute difference
            loss = 0.
            count = 0
            for c in range(len(target)):
                if target[c].numel() > 0 and inputs[c].numel() > 0:
                    loss += torch.abs(torch.norm(inputs[c], dim=1) - (target[c]+self.delta_norm)).mean()
                    count += 1

            if count > 1:
                if torch.is_tensor(loss):
                    return loss/count
                else:
                    return torch.tensor([0.], requires_grad=True, device=inputs[0].device)
            else:
                if torch.is_tensor(loss):
                    return loss
                else:
                    return torch.tensor([0.], requires_grad=True, device=inputs[0].device)
        elif self.ntype == "multiplicative":
            # single, global target norm
            # absolute difference
            # multiplicative delta
            norms = torch.norm(inputs, dim=1)
            deltas = norms - (target*self.delta_norm).expand_as(norms)
            loss = torch.mean(torch.abs(deltas))

            return loss
        else:
            raise ValueError("Invalid norm type, must be: (global, percent, class, multiplicative)")

class perpendicularity(nn.Module):
    def __init__(self):
        super(perpendicularity, self).__init__()
        self.sim = nn.CosineSimilarity()
        return

    def forward(self, inputs): # inputs is a list of class centroids
        loss = 0.
        count = 0

        order = np.random.permutation(len(inputs))
        for i in range(len(inputs)):
            if inputs[order[i]].numel() > 0:
                pivot = inputs[order[i]].detach() # fix a pivot, and shift the other vectors
                for j in range(i+1, len(inputs)):
                    if inputs[order[j]].numel() > 0:
                        count += 1
                        loss += self.sim(pivot, inputs[order[j]])

        if count > 1:
            if torch.is_tensor(loss):
                return (loss/count).squeeze(0)
            else:
                return torch.tensor([0.], requires_grad=True, device=inputs[0].device)
        else:
            if torch.is_tensor(loss):
                return loss.squeeze(0)
            else:
                return torch.tensor([0.], requires_grad=True, device=inputs[0].device)


class CosineSimilarityLoss(nn.Module):
    def __init__(self, dim=1, eps=1e-08):
        super(CosineSimilarityLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=dim, eps=eps)
        self.eps = eps

    def forward(self, inputs, target):
        scores = self.cos(inputs, target)
        return 1. - torch.abs(scores).mean()

# inputs is a list of tensors [(N1,2048), (N2,2048), (N3,2048), ...]
# target is a list of same length
class clustering(nn.Module):
    def __init__(self, type='euclidean'):
        super(clustering, self).__init__()
        self.type = type
        if type == 'euclidean' or type == 'percent':
            self.sim = nn.MSELoss()
        elif type == 'absolute':
            self.sim = nn.L1Loss()
        else:
            self.sim = CosineSimilarityLoss()
        return

    def forward(self, inputs, target):
        loss = 0.

        for i in range(len(inputs)):
            #print(target[i].size(), inputs[i].size())
            if target[i].numel() > 0 and inputs[i].numel() > 0:
                t = self.sim(inputs[i], target[i].detach().expand_as(inputs[i]))
                if self.type == 'percent':
                    loss += t/target.detach().norm() # fix the target as the pivot
                else:
                    loss += t

        if len(inputs) > 1:
            if torch.is_tensor(loss):
                return (loss/len(inputs)).squeeze(0)
            else:
                return torch.tensor([0.], requires_grad=True, device=inputs[0].device)
        else:
            if torch.is_tensor(loss):
                return loss.squeeze(0)
            else:
                return torch.tensor([0.], requires_grad=True, device=inputs[0].device)

class vectorsExtractor(nn.Module):
    def __init__(self, num_classes=19):
        super(vectorsExtractor, self).__init__()
        self.num_classes = num_classes
        return

    def forward(self, feats, y_down):
        # compute centroids and feature vectors sets
        f_c = []
        b_c = []
        n_c = []
        chs = feats.shape[1]
        for c in range(self.num_classes):
            # initialize the cumulative vector list
            t = torch.zeros((0,chs), device=feats.device)
            for b in range(feats.size()[0]):
                mask = y_down[b,...]==c
                t = torch.cat([t, torch.transpose(feats[b,:,mask],0,1)], 0)

            # append the feature vectors to the list
            f_c.append(t.clone())

            # do the same for centroids and norms
            if t.numel() > 0:
                b_c.append(t.mean(dim=0, keepdim=True).clone())
                n_c.append(t.detach().norm(dim=1).mean())
            else:
                b_c.append(t.clone())
                n_c.append(t.detach().norm(dim=1))

        return f_c, b_c, n_c


def predHistDown(preds, shape, hist_th, num_classes, conf_th):
    preds = F.softmax(preds.detach(), dim=1)
    confs, preds = torch.max(preds, dim=1)
    conf_map = F.adaptive_avg_pool2d(confs, (shape[1],shape[0])) # average downsampling
    #confs, preds = np.array(confs.to('cpu')), np.array(preds.to('cpu'), dtype=int)
    preds, conf_map = np.array(preds.to('cpu'), dtype=int), np.array(conf_map.to('cpu'))

    down_preds = []
    for b in range(preds.shape[0]):
        down_preds.append(
            np.expand_dims(
                histDown(preds[b,...],
                  shape,
                  thresh=hist_th,
                  num_classes=num_classes,
                  conf_map=conf_map[b,...],
                  conf_th=conf_th
                ),
            0)
        )

    return torch.from_numpy(np.concatenate(down_preds).astype(np.long))
    
def predNNDown(preds, shape, conf_th):
    preds = F.softmax(preds.detach(), dim=1)
    confs, preds = torch.max(preds, dim=1)
    conf_map = F.adaptive_avg_pool2d(confs, (shape[1],shape[0])) # average downsampling
    mask = conf_map < conf_th
    mask = mask.cpu().numpy()
    preds = np.array(preds.to('cpu'), dtype=int)

    down_preds = []
    for b in range(preds.shape[0]):
        im = Image.fromarray(preds[b]).resize(shape, Image.NEAREST)
        im = np.array(im)
        im[mask[b]] = -1
        down_preds.append(np.expand_dims(im, 0))

    return torch.from_numpy(np.concatenate(down_preds).astype(np.long))

# proto distance
def predPDDown(feats, proto, double_pass=True, conf_th=None, log_twopass=False):
    
    down_preds = -torch.ones((feats.shape[0],)+feats.shape[2:], dtype=torch.long)
    if double_pass and log_twopass:
        old = down_preds.clone()
        old_filter = down_preds.clone()
        new = down_preds.clone()
    
    clean_proto = []
    clean_ids = []
    for c, p in enumerate(proto):
        if p.numel() > 0: 
            clean_ids.append(c)
            #print(p.shape)
            clean_proto.append(p.detach().squeeze(0).unsqueeze(1))
    clean_proto = torch.cat(clean_proto, dim=1).unsqueeze(1)
    clean_ids = torch.Tensor(clean_ids)
    chs = feats.shape[1]
    for b in range(feats.shape[0]):
    
        f = feats[b,...].reshape(chs,-1,1)
        #print(clean_proto.shape, feats.shape, f.shape)
        diff = f-clean_proto
        dist = diff.norm(dim=0)
        pmap = torch.softmax(-dist, dim=1)
        prob, pred = torch.max(pmap, dim=1)
        pred = clean_ids[pred] # map ids back
        
        if double_pass and log_twopass:
            old[b,...] = pred.reshape(feats.shape[2:]).cpu()
        
        if conf_th is not None:
            pred[prob<conf_th] = -1
            
        if double_pass and log_twopass:
            old_filter[b,...] = pred.reshape(feats.shape[2:]).cpu()
        
        if double_pass:
            new_centers = []
            for c in clean_ids:
                new_centers.append(f[:,pred==c].mean(dim=1))
            new_centers = torch.cat(new_centers, dim=1).unsqueeze(1)
            diff = f-clean_proto
            dist = diff.norm(dim=0)
            pmap = torch.softmax(-dist, dim=1)
            prob, pred = torch.max(pmap, dim=1)
            
        if double_pass and log_twopass:
            new[b,...] = pred.reshape(feats.shape[2:]).cpu()
            
        if conf_th is not None:
            pred[prob<conf_th] = -1
            
        down_preds[b,...] = pred.reshape(feats.shape[2:]).cpu()
        if double_pass and log_twopass:
            return down_preds, (old, old_filter, new)
        else:
            return down_preds
        

class IW_MaxSquareloss(nn.Module):
    def __init__(self, ignore_index=-1, num_class=19, ratio=0.2):
        super().__init__()
        self.ignore_index = ignore_index
        self.num_class = num_class
        self.ratio = ratio

    def forward(self, pred, prob, label=None):
        """
        :param pred: predictions (N, C, H, W)
        :param prob: probability of pred (N, C, H, W)
        :param label(optional): the map for counting label numbers (N, C, H, W)
        :return: maximum squares loss with image-wise weighting factor
        """
        # prob -= 0.5
        N, C, H, W = prob.size()
        mask = (prob != self.ignore_index)
        maxpred, argpred = torch.max(prob, 1)
        mask_arg = (maxpred != self.ignore_index)
        argpred = torch.where(mask_arg, argpred, torch.ones(1).to(prob.device, dtype=torch.long) * self.ignore_index)
        if label is None:
            label = argpred
        weights = []
        batch_size = prob.size(0)
        for i in range(batch_size):
            hist = torch.histc(label[i].cpu().data.float(),
                               bins=self.num_class + 1, min=-1,
                               max=self.num_class - 1).float()
            hist = hist[1:]
            weight = \
            (1 / torch.max(torch.pow(hist, self.ratio) * torch.pow(hist.sum(), 1 - self.ratio), torch.ones(1))).to(
                argpred.device)[argpred[i]].detach()
            weights.append(weight)
        weights = torch.stack(weights, dim=0).unsqueeze(1)
        mask = mask_arg.unsqueeze(1).expand_as(prob)
        prior = torch.mean(prob, (2, 3), True).detach()
        loss = -torch.sum((torch.pow(prob, 2) * weights)[mask]) / (batch_size * self.num_class)
        return loss
