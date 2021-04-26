import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from smooth_functions import SAT, ReLU, Swish
"""
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1


    https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/resnet.py
"""

# This is an altered version of this implementation of the ResNet network for cifar images. # From https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py

def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, advProp=False, activation=nn.ReLU()):
        super(BasicBlock, self).__init__()
        self.activationFunc = activation
        self.advProp = advProp # True or False
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)
        self.shortcut = nn.Sequential()

        if self.advProp == True : 
          self.bn1_adv = nn.BatchNorm2d(out_channels)
          self.bn2_adv = nn.BatchNorm2d(out_channels * self.expansion)

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
    
    def forward(self, x, advprop_samples=False):
        # advprop_samples is a boolean a value to know if the current passed batch is made of adversarial examples. 
        if self.advProp == True and advprop_samples == True : 
          out = self.activationFunc.apply(self.bn1_adv(self.conv1(x)))
          out = self.bn2_adv(self.conv2(out))
          out += self.shortcut(x)
          out = self.activationFunc.apply(out)
        else: 
          out = self.activationFunc.apply(self.bn1(self.conv1(x)))
          out = self.bn2(self.conv2(out))
          out += self.shortcut(x)
          out = self.activationFunc.apply(out)
        return out

class AdditionSequential(nn.Sequential):
    def forward(self, x, advprop_samples=False):
        for module in self._modules.values():
            x = module(x, advprop_samples)
        return x


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, adversarialTraining=False, advProp=False, activation=nn.ReLU()):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.advProp = advProp 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.activationFunc = activation
        self.adversarialTraining = adversarialTraining

        # Check to see if the current model has advprop, if so adjust the model with a new batchnormalization layer that will keep separate statistics. 
        if self.advProp == True : 
          self.bn1_adv = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], self.advProp, self.activationFunc, stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], self.advProp, self.activationFunc, stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], self.advProp, self.activationFunc, stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], self.advProp, self.activationFunc, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, out_channels, num_blocks, advProp, activation, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, advProp, activation))
            self.in_channels = out_channels * block.expansion

        return AdditionSequential(*layers)

    def forward(self, x, y, advSamples=False):

      if self.advProp == True and advSamples == True:
        # AdvProp Adversarial Training. 
        out = self.activationFunc.apply(self.bn1_adv(self.conv1(x)))
        out = self.layer1(out, advprop_samples=True)
        out = self.layer2(out, advprop_samples=True)
        out = self.layer3(out, advprop_samples=True)
        out = self.layer4(out, advprop_samples=True)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

      else: 
        # Normal Training without Adversarial Samples. 
        out = self.activationFunc.apply(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

      return out


def resnet18(adversarialTraining=False, advProp=False, activation=ReLU(), labelSmoothing=False):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], adversarialTraining=adversarialTraining, advProp=advProp, activation=activation)

def resnet34(adversarialTraining=False, advProp=False, activation=ReLU(), labelSmoothing=False):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], adversarialTraining=adversarialTraining, advProp=advProp, activation=activation)
