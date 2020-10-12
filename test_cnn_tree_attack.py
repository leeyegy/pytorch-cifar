'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from loss import *
from advertorch.attacks import GradientSignAttack,LinfPGDAttack
from advertorch.context import ctx_noparamgrad_and_eval
from attack.PGDAttack import TreeAttack


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument("--net",default="ResNet18",type=str)
parser.add_argument("--epsilon",type=float,default=0.03137)
parser.add_argument("--beta",type=float,default=0.5)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model dict
net_dict = {"VGG19":VGG('VGG19'),
            "ResNet18": ResNet18(),
            "PreActResNet18": PreActResNet18(),
            "GoogLeNet":GoogLeNet(),
            "DenseNet121":DenseNet121(),
            "ResNeXt29_2x64d":ResNeXt29_2x64d(),
            "MobileNet":MobileNet(),
            "MobileNetV2":MobileNetV2(),
            "DPN92": DPN92(),
            # "ShuffleNetG2":ShuffleNetG2(),
            "SENet18":SENet18(),
            "ShuffleNetV2":ShuffleNetV2(1),
            "EfficientNetB0":EfficientNetB0(),
            "RegNetX_200MF":RegNetX_200MF(),
            "WideResNet": WideResNet()
            }

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(
    root='/data/liyanjie/.torch/datasets', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='/data/liyanjie/.torch/datasets', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model 10 classifier
print('==> Building model..')
net = net_dict[args.net]
net = net.to(device)

# net_6 = ResNet18(num_classes=6)
net_6 = WideResNet()
net_6 = net_6.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    net_6 = torch.nn.DataParallel(net_6)
    cudnn.benchmark = True

# resume
checkpoint = torch.load(os.path.join("checkpoint","mart_WideResNet","beta_6","ckpt.pth"))
net.load_state_dict(checkpoint['net'])
start_epoch = checkpoint['epoch']

checkpoint = torch.load(os.path.join("checkpoint","mart_6_WideResNet","30epoch_whole_234567_beta_6.0","ckpt.pth"))
net_6.load_state_dict(checkpoint['net'])

criterion = nn.CrossEntropyLoss()

adversary = TreeAttack(main_model=net,attached_model=net_6,eps=args.epsilon,perturb_steps=20,step_size=args.epsilon/10,beta=args.beta)

def gen_ori_tree_prediction(data):
    # set state
    net.eval()
    net_6.eval()

    # generate ori prediction
    with torch.no_grad():
        output = net(data)
        ori_prediction = torch.max(output,1)[1]

    # generate tree prediction
    ## get reset data
    tree_prediction = ori_prediction.clone().detach()
    mask = (ori_prediction<8) & (ori_prediction>1) # 2 3 4 5 6 7
    retest_data = data[mask].clone().detach()

    with torch.no_grad():
        output = net_6(retest_data)
        tree_prediction[mask] = torch.max(output,1)[1] + 2

    return ori_prediction,tree_prediction

def test():
    total = 0
    ori_cln_correct = 0
    ori_adv_correct = 0
    tree_cln_correct = 0
    tree_adv_correct = 0

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)

        # get clean prediction
        ori_cln_prediction,tree_cln_prediction = gen_ori_tree_prediction(inputs)
        ori_cln_correct += ori_cln_prediction.eq(targets.view_as(ori_cln_prediction)).sum().item()
        tree_cln_correct += tree_cln_prediction.eq(targets.view_as(tree_cln_prediction)).sum().item()

        with ctx_noparamgrad_and_eval(net):
            pgd_data = adversary.perturb(inputs.clone().detach(), targets)
        ori_adv_prediction,tree_adv_prediction = gen_ori_tree_prediction(pgd_data)
        ori_adv_correct += ori_adv_prediction.eq(targets.view_as(ori_adv_prediction)).sum().item()
        tree_adv_correct += tree_adv_prediction.eq(targets.view_as(tree_adv_prediction)).sum().item()

        progress_bar(batch_idx, len(testloader), 'Ori: clean acc: %.3f%% (%d/%d) adv acc:  %.3f%% (%d/%d)| tree: clean acc: %.3f%% (%d/%d) adv acc:  %.3f%% (%d/%d)'
                     % ( 100.*ori_cln_correct/total, ori_cln_correct, total,
                         100.*ori_adv_correct/total,ori_adv_correct,total,
                         100.*tree_cln_correct/total,tree_cln_correct,total,
                         100.*tree_adv_correct/total,tree_adv_correct,total))

test()
