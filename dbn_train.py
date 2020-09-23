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
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument("--net",default="ResNet18",type=str)
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument("--min_loss",type=str,default="CE",choices=["CE","CS","FOCAL"])
parser.add_argument("--max_loss",type=str,default="CE",choices=["CE","CS","FOCAL"])
parser.add_argument("--attack_method",type=str,default="FGSM")
parser.add_argument("--epsilon",type=float,default=0.03137)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
save_path = os.path.join("checkpoint",args.min_loss+"_"+args.max_loss+"_"+args.net,"adv_"+args.attack_method+"_"+str(args.epsilon))
if not os.path.exists(save_path):
    os.makedirs(save_path)

# model dict
net_dict = {"VGG19":VGG('VGG19'),
            "ResNet18": ResNet18(),
            "ResNet18_cosine":ResNet18_cosine(),
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
            "ResNet18_dbn":ResNet18_dbn()
}

loss_dict = {"CE":nn.CrossEntropyLoss() ,
            "CS": Cosine_Similarity_Loss(),
            "FOCAL":Focal_Loss()
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
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(save_path,'ckpt.pth'))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


criterion = loss_dict[args.min_loss]
adversary_loss = loss_dict[args.max_loss]
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)


#define adversary
if args.attack_method == "FGSM":
    adversary = GradientSignAttack(net,eps=args.epsilon,loss_fn=adversary_loss,clip_min=0.0,clip_max=1.0)
elif args.attack_method == "PGD":
    adversary = LinfPGDAttack(net,eps=args.epsilon,nb_iter=10,eps_iter=0.007,loss_fn=adversary_loss,rand_init=True)
PGD_adversary = LinfPGDAttack(net,eps=0.03137,nb_iter=10,eps_iter=0.007,loss_fn=adversary_loss,rand_init=True)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)


        optimizer.zero_grad()
        cln_outputs = net(inputs,mode="cln")
        with ctx_noparamgrad_and_eval(net):
            adv_data = adversary.perturb(inputs,targets)
        adv_outputs = net(adv_data,mode="adv")
        loss = criterion(cln_outputs, targets) + criterion(adv_outputs,targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        total += targets.size(0)

        correct += get_correct_num(cln_outputs,targets,args.max_loss)

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
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
        correct += get_correct_num(outputs,targets,args.max_loss)

        with ctx_noparamgrad_and_eval(net):
            pgd_data = PGD_adversary.perturb(inputs.clone().detach(), targets)
        with torch.no_grad():
            outputs = net(pgd_data,mode="adv")
        loss = criterion(outputs, targets)
        pgd_loss += loss.item()
        pgd_correct += get_correct_num(outputs,targets,args.max_loss)

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) PgdAcc:%.3f%% (%d/%d)'
                     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total,100.*pgd_correct/total,pgd_correct,total))
    # Save checkpoint.
    acc = 100.*pgd_correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, os.path.join(save_path,'ckpt.pth'))
        best_acc = acc
    if epoch == 119:
        print('Saving Last..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, os.path.join(save_path, 'ckpt_last.pth'))


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

for epoch in range(start_epoch, 120):
    adjust_learning_rate(optimizer,epoch)
    train(epoch)
    test(epoch)