from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from models import *
from utils import  progress_bar
from loss import *
import numpy as np
import time
from advertorch.attacks import GradientSignAttack,LinfPGDAttack
from advertorch.context import ctx_noparamgrad_and_eval
from tensorboardX import SummaryWriter
from util.analyze_easy_hard import _analyze_correct_class_level,_average_output_class_level,_calculate_information_entropy
from autoattack import AutoAttack



parser = argparse.ArgumentParser(description='PyTorch CIFAR MART Defense')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=120, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=3.5e-3,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.03137,type=float,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,
                    help='perturb step size')
parser.add_argument('--beta', default=5.0,
                    help='weight before kl (misclassified examples)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument("--net",default="ResNet18",type=str)
parser.add_argument('--resume_best', default=False, action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--resume_last', default=False, action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

# for wideres
if args.net == "WideResNet":
    args.weight_decay = 7e-4
    args.lr = 0.1

torch.manual_seed(args.seed)
device ='cuda' if torch.cuda.is_available() else 'cpu'
kwargs = {'num_workers': 10, 'pin_memory': True}
torch.backends.cudnn.benchmark = True

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
save_path = os.path.join("checkpoint","mart_6_"+args.net,"50epoch_whole_234567_beta_"+str(args.beta))
exp_name = os.path.join("runs", "mart_6_"+args.net,"50epoch_whole_234567_beta_"+str(args.beta))
# save_path = os.path.join("checkpoint","mart_6_"+args.net,"234567_beta_"+str(args.beta))
# exp_name = os.path.join("runs", "mart_6_"+args.net,"234567_beta_"+str(args.beta))

if not os.path.exists(save_path):
    os.makedirs(save_path)
# define tensorboard
writer = SummaryWriter(exp_name)

# model dict
net_dict = {"VGG19":VGG('VGG19'),
            "ResNet18": ResNet18(num_classes=6),
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
            "WideResNet":WideResNet()
}

# Model
print('==> Building model..')
net = net_dict[args.net]
net = net.to(device)

# Data
print('==> Preparing data..')
# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
trainset = torchvision.datasets.CIFAR10(root='/data/liyanjie/.torch/datasets/', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=10)
testset = torchvision.datasets.CIFAR10(root='/data/liyanjie/.torch/datasets/', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=10)

if device == 'cuda':
    net = torch.nn.DataParallel(net)

# init
model_dict = net.state_dict()
checkpoint = torch.load(os.path.join("checkpoint","mart_WideResNet","beta_6","ckpt.pth"))
pretrained_dict = checkpoint['net']
pretrained_dict = {k: v for k, v in pretrained_dict.items() if not ("linear" in k )}
model_dict.update(pretrained_dict)
net.load_state_dict(model_dict)

assert not (args.resume_best and args.resume_last)

if args.resume_best:
    # Load checkpoint.
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    print("best resumed")
    checkpoint = torch.load(os.path.join(save_path,'ckpt.pth'))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print('==> Resuming from checkpoint {}'.format(start_epoch))

if args.resume_last:
    # Load checkpoint.
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(save_path,'ckpt_last.pth'))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print('==> Resuming from checkpoint {}'.format(start_epoch))

#define adversary
PGD_adversary = LinfPGDAttack(net,eps=args.epsilon,nb_iter=20,eps_iter=args.epsilon/10,loss_fn=nn.CrossEntropyLoss(),rand_init=True)
# AA_adversary = AutoAttack(net, norm='Linf', eps=args.epsilon, version='standard')

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # filter out
        mask = (target<8) & (target>1)
        data = data[mask]
        target = target[mask]
        target -= 2

        optimizer.zero_grad()

        # calculate robust loss
        loss = mart_loss(model=model,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           step_size=args.step_size,
                           epsilon=args.epsilon,
                           perturb_steps=args.num_steps,
                           beta=args.beta)
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def test(epoch):
    global best_acc
    net.eval()
    pgd_correct = 0
    correct = 0
    total = 0

    adv_stat_correct = torch.zeros([6]).cuda()
    adv_stat_total = torch.zeros([6]).cuda()

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # filter out
        mask = (targets<8) & (targets>1)
        inputs = inputs[mask]
        targets = targets[mask]
        targets -= 2

        with torch.no_grad():
            outputs = net(inputs)
        total += targets.size(0)
        correct += get_correct_num(outputs,targets,"CE")

        with ctx_noparamgrad_and_eval(net):
            pgd_data = PGD_adversary.perturb(inputs.clone().detach(), targets)
        # pgd_data = AA_adversary.run_standard_evaluation(inputs, targets, bs=args.test_batch_size)

        with torch.no_grad():
            outputs = net(pgd_data)
        pgd_correct += get_correct_num(outputs,targets,"CE")

        # for tensorboard
        prediction = outputs.max(1,keepdim=True)[1].view_as(targets)
        _analyze_correct_class_level(prediction, targets, adv_stat_correct, adv_stat_total)


        progress_bar(batch_idx, len(test_loader), '| Acc: %.3f%% (%d/%d) PgdAcc:%.3f%% (%d/%d)'
                     % (100.*correct/total, correct, total,100.*pgd_correct/total,pgd_correct,total))

    adv_stat_correct = 100.0 * adv_stat_correct / adv_stat_total

    #monitor acc - class level
    writer.add_scalars("test_adv_acc_class_level",{str(i): adv_stat_correct[i] for i in range(4)},epoch)

    #monitor acc - class level
    writer.add_scalar("test_adv_acc",100.*pgd_correct/total,epoch)

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
        torch.save(state, os.path.join(save_path,'ckpt.pth'.format(epoch)))
        best_acc = acc
    if epoch == args.epochs-1:
        print('Saving Last..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, os.path.join(save_path, 'ckpt-{}.pth'.format(epoch)))

# def adjust_learning_rate(optimizer, epoch):
#     """decrease the learning rate"""
#     lr = args.lr
#     if epoch >= 25:
#         lr = args.lr * 0.001
#     elif epoch >= 15:
#         lr = args.lr * 0.01
#     elif epoch >= 10:
#         lr = args.lr * 0.1
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#
def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 50:
        lr = args.lr * 0.001
    elif epoch >= 40:
        lr = args.lr * 0.01
    elif epoch >= 30:
        lr = args.lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# def adjust_learning_rate(optimizer, epoch):
#     """decrease the learning rate"""
#     lr = args.lr
#     if epoch >= 100:
#         lr = args.lr * 0.001
#     elif epoch >= 90:
#         lr = args.lr * 0.01
#     elif epoch >= 75:
#         lr = args.lr * 0.1
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)

for epoch in range(start_epoch, args.epochs):
    adjust_learning_rate(optimizer,epoch)
    train(args, net, device, train_loader, optimizer, epoch)
    test(epoch)
    writer.close()
