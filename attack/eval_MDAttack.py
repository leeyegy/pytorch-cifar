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

import  sys
sys.path.append("../")
from models import *
from utils import progress_bar
from loss import *
from MDAttack import *


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument("--net",default="ResNet18",type=str)
parser.add_argument('--test_model_path', type=str)
parser.add_argument("--epsilon",type=float,default=0.03137)
# attack type
parser.add_argument('--mdmt', action='store_true', default=False)
parser.add_argument('--md', action='store_true', default=False)

parser.add_argument('--num-steps', default=40, type=int,
                    help='perturb number of steps')
parser.add_argument('--first-step-size', default=16. / 255., type=float,
                    help='perturb step size for first stage')
parser.add_argument('--step-size', default=2. / 255., type=float,
                    help='perturb step size for second stage')
parser.add_argument('--random',
                    type=int,
                    default=1,
                    help='number of random initialization for PGD')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy

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
            "RegNetX_200MF":RegNetX_200MF()
}

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='/home/Leeyegy/.torch/datasets', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='/home/Leeyegy/.torch/datasets', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')

net = net_dict[args.net]
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    # cudnn.benchmark = True

checkpoint = torch.load(os.path.join(args.test_model_path))
net.load_state_dict(checkpoint['net'])
start_epoch = checkpoint['epoch']

def eval_adv_test_whitebox(model, device, test_loader, vmin=0.0, vmax=1.0):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_correct_total = 0
    natural_correct_total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        X, y = Variable(data, requires_grad=True), Variable(target)

        if args.md:
            nat_correct, adv_correct, X_adv = MD_attack(model, X, y, epsilon=args.epsilon,
                                                        num_steps=args.num_steps,
                                                        step_size=args.step_size,
                                                        num_random_starts=args.random,
                                                        v_min=vmin, v_max=vmax,
                                                        change_point=args.num_steps / 2,
                                                        first_step_size=args.first_step_size)
        elif args.mdmt:
            nat_correct, adv_correct, X_adv = MDMT_attack(model, X, y, epsilon=args.epsilon,
                                                          num_steps=args.num_steps,
                                                          step_size=args.step_size,
                                                          v_min=vmin, v_max=vmax,
                                                          change_point=args.num_steps / 2,
                                                          first_step_size=args.first_step_size)

        pertub = X_adv - X.detach().cpu().clone().numpy()
        valid = (pertub <= args.epsilon * (vmax - vmin) + 1e-6) & (
                pertub >= -args.epsilon * (vmax - vmin) - 1e-6)
        assert np.all(valid), 'perturb outrange!'
        num_adv_correct = adv_correct.float().sum().item()
        # print('adv correct (white-box): ', num_adv_correct)
        natural_correct_total += nat_correct.float().sum().item()
        robust_correct_total += num_adv_correct

    print('natural_correct_total: ', natural_correct_total)
    print('robust_correct_total: ', robust_correct_total)

eval_adv_test_whitebox(net,device,testloader,0.0,1.0)