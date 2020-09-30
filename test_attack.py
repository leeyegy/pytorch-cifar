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


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument("--net",default="ResNet18",type=str)
parser.add_argument('--test_model_path', type=str)
parser.add_argument("--loss",type=str,default="CE",choices=["CE","CS"])
parser.add_argument("--attack_method",type=str,default="FGSM",choices=["PGD","FGSM"])
parser.add_argument("--epsilon",type=float,default=0.03137)

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
checkpoint = torch.load(os.path.join(args.test_model_path))
net.load_state_dict(checkpoint['net'])
start_epoch = checkpoint['epoch']

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# if args.loss == "CS":
#     checkpoint = torch.load(os.path.join(args.test_model_path))
#     net.load_state_dict(checkpoint['net'])
# elif args.loss == "CE":
#     checkpoint = torch.load(os.path.join(args.test_model_path))
#     net.load_state_dict(checkpoint['net'])
#     # net.load_state_dict(torch.load(os.path.join(args.test_model_path)))
# best_acc = checkpoint['acc']
# # start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss() if args.loss == "CE" else Cosine_Similarity_Loss()



#define adversary
if args.attack_method == "FGSM":
    adversary = GradientSignAttack(net,eps=args.epsilon,loss_fn=criterion,clip_min=0.0,clip_max=1.0)
elif args.attack_method == "PGD":
    adversary = LinfPGDAttack(net,eps=args.epsilon,nb_iter=10,eps_iter=0.007,loss_fn=criterion,rand_init=True)


def test():
    global best_acc
    net.eval()
    test_loss = 0
    pgd_loss = 0
    pgd_correct = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            outputs = net(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
        total += targets.size(0)
        correct += get_correct_num(outputs,targets,args.loss)

        with ctx_noparamgrad_and_eval(net):
            pgd_data = adversary.perturb(inputs.clone().detach(), targets)
        with torch.no_grad():
            outputs = net(pgd_data)
        loss = criterion(outputs, targets)
        pgd_loss += loss.item()
        pgd_correct += get_correct_num(outputs,targets,args.loss)

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) PgdAcc:%.3f%% (%d/%d)'
                     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total,100.*pgd_correct/total,pgd_correct,total))

test()
