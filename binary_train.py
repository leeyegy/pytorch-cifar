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
from tensorboardX import SummaryWriter
from util.analyze_easy_hard import _analyze_correct_class_level,_average_output_class_level,_calculate_information_entropy

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument("--net",default="ResNet18",type=str)
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument("--min_loss",type=str,default="CE",choices=["CE","CS","FOCAL","SPLoss","FOCAL_INDI","Ban_Loss","Easy2hardLoss"])
parser.add_argument("--max_loss",type=str,default="CE",choices=["CE","CS","FOCAL"])
parser.add_argument("--attack_method",type=str,default="FGSM")
parser.add_argument("--epsilon",type=float,default=0.03137)
parser.add_argument("--gamma",type=float,default=2.0)
parser.add_argument("--pick_up",type=int,default=0)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
save_path = os.path.join("checkpoint","binary",args.min_loss+"_"+args.max_loss+"_"+args.net,str(args.pick_up)) if args.min_loss != "FOCAL_INDI" else  os.path.join("checkpoint","binary",args.min_loss+"_"+str(args.gamma)+"_"+args.max_loss+"_"+args.net,str(args.pick_up))
if not os.path.exists(save_path):
    os.makedirs(save_path)

# define tensorboard
exp_name = os.path.join("runs", "binary",args.min_loss+"_"+args.max_loss+"_"+args.net,str(args.pick_up)) if args.min_loss != "FOCAL_INDI" else os.path.join("runs", "binary",args.min_loss+"_"+str(args.gamma)+"_"+args.max_loss+"_"+args.net,str(args.pick_up))
writer = SummaryWriter(exp_name)

# model dict
net_dict = {
            "ResNet18": ResNet18(num_classes=2),
}

loss_dict = {"CE":nn.CrossEntropyLoss() ,
            "CS": Cosine_Similarity_Loss(),
            "FOCAL":Focal_Loss(),
            "SPLoss":SPLoss(),
            "FOCAL_INDI":Focal_Loss(individual=True,gamma=args.gamma),
            "Ban_Loss":Ban_Loss(),
            "Easy2hardLoss" : Easy2hardLoss()
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
if args.net != "ResNet18_cosine":
    criterion.mode = "normal"
adversary_loss = loss_dict[args.max_loss]
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)


#define adversary
if args.attack_method == "FGSM":
    adversary = GradientSignAttack(net,eps=args.epsilon,loss_fn=adversary_loss,clip_min=0.0,clip_max=1.0)
elif args.attack_method == "PGD":
    adversary = LinfPGDAttack(net,eps=args.epsilon,nb_iter=10,eps_iter=0.007,loss_fn=adversary_loss,rand_init=True)
PGD_adversary = LinfPGDAttack(net,eps=0.03137,nb_iter=20,eps_iter=0.007,loss_fn=adversary_loss,rand_init=True)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    adv_stat_correct = torch.zeros([2]).cuda()
    adv_stat_total = torch.zeros([2]).cuda()
    adv_stat_output = torch.zeros([2, 2]).cuda()
    adv_stat_shannon_total = torch.zeros([2]).cuda()

    for batch_idx, (inputs, targets_ori) in enumerate(trainloader):
        inputs, targets_ori = inputs.to(device), targets_ori.to(device)
        targets = targets_ori.clone().detach()
        targets[targets_ori==args.pick_up] = 1
        targets[targets_ori!=args.pick_up] = 0

        optimizer.zero_grad()
        with ctx_noparamgrad_and_eval(net):
            adv_data = adversary.perturb(inputs,targets)
        outputs = net(adv_data)

        # for tensorboard
        prediction = outputs.max(1,keepdim=True)[1].view_as(targets)
        _analyze_correct_class_level(prediction, targets, adv_stat_correct, adv_stat_total)
        _average_output_class_level(F.softmax(outputs,dim=1), targets, adv_stat_output, adv_stat_shannon_total)

        loss = criterion(outputs, targets) if args.min_loss != "SPLoss" and args.min_loss !="Easy2hardLoss" else criterion(outputs, targets,epoch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        total += targets.size(0)

        correct += get_correct_num(outputs,targets,args.max_loss)

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    adv_stat_correct = 100.0 * adv_stat_correct / adv_stat_total
    adv_stat_output /= adv_stat_shannon_total
    adv_entropy = _calculate_information_entropy(adv_stat_output)

    #monitor shannon - class level
    writer.add_scalars("train_adv_shannon_class_level",{str(i): adv_entropy[i] for i in range(2)},epoch)

    #monitor acc - class level
    writer.add_scalars("train_adv_acc_class_level",{str(i): adv_stat_correct[i] for i in range(2)},epoch)

def test(epoch):
    global best_acc
    net.eval()
    pgd_correct = 0
    correct = 0
    total = 0

    adv_stat_correct = torch.zeros([2]).cuda()
    adv_stat_total = torch.zeros([2]).cuda()
    adv_stat_output = torch.zeros([2, 2]).cuda()
    adv_stat_shannon_total = torch.zeros([2]).cuda()

    for batch_idx, (inputs, targets_ori) in enumerate(testloader):
        inputs, targets_ori = inputs.to(device), targets_ori.to(device)
        targets = targets_ori.clone().detach()
        targets[targets_ori==args.pick_up] = 1
        targets[targets_ori!=args.pick_up] = 0

        with torch.no_grad():
            outputs = net(inputs)
        total += targets.size(0)
        correct += get_correct_num(outputs,targets,args.max_loss)

        with ctx_noparamgrad_and_eval(net):
            pgd_data = PGD_adversary.perturb(inputs.clone().detach(), targets)
        with torch.no_grad():
            outputs = net(pgd_data)
        pgd_correct += get_correct_num(outputs,targets,args.max_loss)

        # for tensorboard
        prediction = outputs.max(1,keepdim=True)[1].view_as(targets)
        _analyze_correct_class_level(prediction, targets, adv_stat_correct, adv_stat_total)
        _average_output_class_level(F.softmax(outputs,dim=1), targets, adv_stat_output, adv_stat_shannon_total)


        progress_bar(batch_idx, len(testloader), '| Acc: %.3f%% (%d/%d) PgdAcc:%.3f%% (%d/%d)'
                     % (100.*correct/total, correct, total,100.*pgd_correct/total,pgd_correct,total))

    adv_stat_correct = 100.0 * adv_stat_correct / adv_stat_total
    adv_stat_output /= adv_stat_shannon_total
    adv_entropy = _calculate_information_entropy(adv_stat_output)

    #monitor shannon - class level
    writer.add_scalars("test_adv_shannon_class_level",{str(i): adv_entropy[i] for i in range(2)},epoch)

    #monitor acc - class level
    writer.add_scalars("test_adv_acc_class_level",{str(i): adv_stat_correct[i] for i in range(2)},epoch)

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
writer.close()
