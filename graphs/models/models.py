import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def DeeplabNW(num_classes, backbone, pretrained=True):
    print('DeeplabV2 is being used with {} as backbone'.format(backbone))
    if backbone.lower() == 'ResNet101'.lower():
    
        model = DeeplabResnet(Bottleneck, [3, 4, 23, 3], num_classes)
        if pretrained:
            restore_from = './pretrained_model/resnet101-5d3b4d8f.pth'
            saved_state_dict = torch.load(restore_from)

            new_params = model.state_dict().copy()
            for i in saved_state_dict:
                i_parts = i.split('.')
                if not i_parts[1] == 'layer5' and not i_parts[0] == 'fc':
                    new_params[i] = saved_state_dict[i]
            model.load_state_dict(new_params)
            
    elif backbone.lower() == 'ResNet50'.lower():
    
        model = DeeplabResnet(Bottleneck, [3, 4, 6, 3], num_classes)
        if pretrained:
            restore_from = './pretrained_model/resnet50-19c8e357.pth'
            saved_state_dict = torch.load(restore_from)

            new_params = model.state_dict().copy()
            for i in saved_state_dict:
                i_parts = i.split('.')
                if not i_parts[1] == 'layer5' and not i_parts[0] == 'fc':
                    new_params[i] = saved_state_dict[i]
            model.load_state_dict(new_params)
            
    elif backbone.lower() == 'VGG16'.lower():
    
        restore_from = './pretrained_model/vgg16-397923af.pth'
        model = DeeplabVGG16(num_classes, restore_from=restore_from, pretrained=pretrained)
        
    elif backbone.lower() == 'VGG13'.lower():
    
        restore_from = './pretrained_model/vgg13-c768596a.pth'
        model = DeeplabVGG13(num_classes, restore_from=restore_from, pretrained=pretrained)
        
    else:
        raise Exception

    return model

# ---------------------------------------------------------
#                          DEEPLAB 
# ---------------------------------------------------------

class DeepLabV2(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(DeepLabV2, self).__init__()
        
        self.conv2d_list = nn.ModuleList()
        
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
            return out
        

# ---------------------------------------------------------
#                           RESNET
# ---------------------------------------------------------

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, bn_momentum=0.1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class DeeplabResnet(nn.Module):
    def __init__(self, block, layers, num_classes):
        
        self.inplanes = 64
        super(DeeplabResnet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        for i in self.bn1.parameters():
            i.requires_grad = False
            
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        
        self.clas = self._make_pred_layer(DeepLabV2, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * 4 or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * 4,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4)
                )
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * 4
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

    def forward(self, x):
        input_size = x.size()[2:]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feats = self.layer4(x)
        
        x = self.clas(feats)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)

        return x, feats

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.clas.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.lr},
                {'params': self.get_10x_lr_params(), 'lr': 10 * args.lr}]
                
# ---------------------------------------------------------
#                            VGG
# ---------------------------------------------------------

class DeeplabVGG16(nn.Module):
    def __init__(self, num_classes, restore_from=None, pretrained=False):
        super(DeeplabVGG16, self).__init__()
        vgg = models.vgg16()
        if pretrained:
            vgg.load_state_dict(torch.load(restore_from))

        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())
        features = nn.Sequential(*(features[i] for i in range(30) if i != 23))

        for i in [23,25,27]:
            features[i].dilation = (2,2)
            features[i].padding = (2,2)

        fc6 = nn.Conv2d(512, 1024, kernel_size=3, padding=4, dilation=4)
        fc7 = nn.Conv2d(1024, 1024, kernel_size=3, padding=4, dilation=4)

        self.features = nn.Sequential(*([features[i] for i in range(len(features))] + [ fc6, nn.ReLU(inplace=True), fc7, nn.ReLU(inplace=True)]))
        self.classifier = DeepLabV2(1024, [6,12,18,24],[6,12,18,24],num_classes)


    def forward(self, x):
        input_size = x.size()[2:]
        feats = self.features(x)
        x = self.classifier(feats)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        return x, feats


    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.features)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.classifier.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i


    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.lr},
                {'params': self.get_10x_lr_params(), 'lr': args.lr}]
                
class DeeplabVGG13(nn.Module):
    def __init__(self, num_classes, restore_from=None, pretrained=False):
        super(DeeplabVGG13, self).__init__()
        vgg = models.vgg13()
        if pretrained:
            vgg.load_state_dict(torch.load(restore_from))

        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())
        features = nn.Sequential(*(features[i] for i in range(24) if i != 19))

        #for i in [17,19,21]:
        for i in [19,21]:
            features[i].dilation = (2,2)
            features[i].padding = (2,2)

        fc6 = nn.Conv2d(512, 1024, kernel_size=3, padding=4, dilation=4)
        fc7 = nn.Conv2d(1024, 1024, kernel_size=3, padding=4, dilation=4)

        self.features = nn.Sequential(*([features[i] for i in range(len(features))] + [ fc6, nn.ReLU(inplace=True), fc7, nn.ReLU(inplace=True)]))
        self.classifier = DeepLabV2(1024, [6,12,18,24],[6,12,18,24],num_classes)


    def forward(self, x):
        input_size = x.size()[2:]
        feats = self.features(x)
        x = self.classifier(feats)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        return x, feats


    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.features)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.classifier.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i


    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.lr},
                {'params': self.get_10x_lr_params(), 'lr': args.lr}]
