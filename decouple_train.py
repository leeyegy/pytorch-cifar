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
from attack.PGDAttack import  *
from loss import  _mart_loss ,_madry_loss



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
parser.add_argument('--beta', default=5.0,type=float,
                    help='weight before kl (misclassified examples)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument("--net",default="Decouple18",type=str)
parser.add_argument('--resume_best', default=False, action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--resume_last', default=False, action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument("--classify-loss",type=str,default="mart")

args = parser.parse_args()

# for wideres
if args.net == "WideResNet":
    args.weight_decay = 7e-4
    args.lr = 0.1

torch.manual_seed(args.seed)
device ='cuda' if torch.cuda.is_available() else 'cpu'
kwargs = {'num_workers': 10, 'pin_memory': True}
torch.backends.cudnn.benchmark = True

best_acc = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
save_path = os.path.join("checkpoint","decouple_"+args.net,"mart_5_weight_beta_"+str(args.beta))
exp_name = os.path.join("runs", "decouple_"+args.net,"mart_5_weight_beta_"+str(args.beta))

if not os.path.exists(save_path):
    os.makedirs(save_path)
# define tensorboard
writer = SummaryWriter(exp_name)

# model dict
net_dict = {"VGG19":VGG('VGG19'),
            "ResNet18_Representation": ResNet18_Representation(),
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
            "WideResNet":WideResNet(),
            "ResNet18":ResNet18(),
            "Decouple18":Decouple18()
}

# loss dict
loss_dict = {"mart":_mart_loss,
             "madry":_madry_loss,}

classifier_loss = loss_dict[args.classify_loss]

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
trainset = torchvision.datasets.CIFAR10(root='/home/Leeyegy/.torch/datasets/', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=10)
testset = torchvision.datasets.CIFAR10(root='/home/Leeyegy/.torch/datasets/', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=10)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
assert not (args.resume_best and args.resume_last)

# init from pre-trained model
# checkpoint = torch.load(os.path.join("checkpoint","mart_ResNet18","beta_5","ckpt.pth"))
# checkpoint = torch.load(os.path.join("checkpoint","decouple_Decouple18","weight_beta_10.0","ckpt.pth"))
# checkpoint = torch.load(os.path.join("checkpoint","decouple_Decouple18","weight_beta_6.0","ckpt.pth"))
# net.load_state_dict(checkpoint['net'])
# print(checkpoint['acc'])

# save_model = checkpoint['net']
# model_dict =  net.state_dict()
# state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
# model_dict.update(state_dict)
# net.load_state_dict(model_dict)

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
# adversary = PGD(model=net,eps=args.epsilon,perturb_steps=args.num_steps,step_size=args.step_size) # for train
# PGD_adversary = PGD(model=net,eps=args.epsilon,perturb_steps=20,step_size=args.epsilon/10) # for test
PGD_adversary = LinfPGDAttack(net,eps=args.epsilon,nb_iter=20,eps_iter=args.epsilon/10,loss_fn=nn.CrossEntropyLoss(),rand_init=True)

# PGD_adversary = PGD(net,eps=args.epsilon,nb_iter=20,eps_iter=args.epsilon/10,loss_fn=nn.CrossEntropyLoss(),rand_init=True)
# AA_adversary = AutoAttack(net, norm='Linf', eps=args.epsilon, version='standard')

# define lr adjusting
def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 100:
        lr = args.lr * 0.001
    elif epoch >= 90:
        lr = args.lr * 0.01
    elif epoch >= 75:
        lr = args.lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def freeze(model, train_mode="classifier"):
    if train_mode == "classifier":
        # 冻结非linear的参数
        for name, param in model.named_parameters():
            param.requires_grad = True
            # if "linear" in name:
            #     param.requires_grad = True
            # else:
            #     param.requires_grad = False
    elif train_mode == "representation":
        # 冻结linear的参数
        for name, param in model.named_parameters():
            if "linear" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    else:
        raise

def train(args, model, device, train_loader, classifier_optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # optimize classifier
        classifier_optimizer.zero_grad()
        loss = weight_penalization_mart_loss(model=model,
                                 x_natural=data,
                                 y=target,
                                 optimizer=classifier_optimizer,
                                 step_size=args.step_size,
                                 epsilon=args.epsilon,
                                 perturb_steps=args.num_steps,
                                 beta=args.beta,
                                 distance='l_inf')
        loss.backward()
        classifier_optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t classifier loss: {:.6f} '.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),loss.item()))
#
# def train(args, model, device, train_loader, classifier_optimizer,representation_optimizer, epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         # generate adv data
#         with ctx_noparamgrad_and_eval(net):
#             adv_data = adversary.perturb(data.clone().detach(), target)
#
#         # optimize representation
#         representation_optimizer.zero_grad()
#         freeze(model,train_mode="representation")
#         loss_ = representation_loss(model,data,adv_data,target)
#         loss_.backward()
#         representation_optimizer.step()
#
#         # optimize classifier
#         classifier_optimizer.zero_grad()
#         freeze(model,train_mode="classifier")
#         loss = classifier_loss(model,data,adv_data,target,args.beta)
#         loss.backward()
#         classifier_optimizer.step()
#
#         # print progress
#         if batch_idx % args.log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\trepresentation loss: {:.6f} classifier loss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                        100. * batch_idx / len(train_loader), loss_.item(),loss.item()))

def test(epoch):
    global best_acc
    net.eval()
    pgd_correct = 0
    correct = 0
    total = 0

    adv_stat_correct = torch.zeros([10]).cuda()
    adv_stat_total = torch.zeros([10]).cuda()

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            outputs = net(inputs)
        total += targets.size(0)
        correct += get_correct_num(outputs,targets,"CE")

        with ctx_noparamgrad_and_eval(net):
            pgd_data = PGD_adversary.perturb(inputs.clone().detach(), targets)

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
    writer.add_scalars("test_adv_acc_class_level",{str(i): adv_stat_correct[i] for i in range(10)},epoch)

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
        torch.save(state, os.path.join(save_path,'ckpt.pth'))
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

# define optimizer
freeze(net,train_mode="classifier")
classifier_optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)

# freeze(net,train_mode="representation")
# representation_optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr,
#                       momentum=args.momentum, weight_decay=args.weight_decay)

for epoch in range(start_epoch, args.epochs):
    # classifier层weight长度正则
    # model_dict = net.state_dict()
    # for name,param in net.named_parameters():
    #     if "linear" in name and "bias" not in name:
    #         # update weight
    #         cls_weight = param.clone().detach()
            # for i in range(cls_weight.size()[0]):
            #     cls_weight[i] /= torch.norm(cls_weight[i])
            # state_dict = {name:cls_weight}
            # model_dict.update(state_dict)
            # net.load_state_dict(model_dict)

            # u,s,v = torch.svd(cls_weight)
            # print(s)
            # # cover = u @ torch.diag(s)
            # # print(cover.size())
            # # cover = cover @ v.T
            # # print(cover.size())
            # # print(cover-cls_weight)
            # # print("对线性层512*10进行奇异值分解的结果的特征值：")
            # new_weight = u[:,0:9] @ torch.diag(s[0:9])
            # new_weight = new_weight @ (v[:,0:9]).T
            # print(new_weight.size())
            #
            # u,s,v = torch.svd(new_weight)
            # print(s)

            # dis = torch.ones([10]).cuda()
            # closest_sim = torch.ones([10]).cuda()
            # for i in range(cls_weight.size()[0]):
            #     print(torch.norm(cls_weight[i]))
            #     for k in range(cls_weight.size()[0]):
            #         dis[k] = 1 - F.cosine_similarity(torch.unsqueeze(param[i],0),torch.unsqueeze(param[k],0))
            #     sorted,index = torch.sort(dis)
            #     print("第{}类与第{}类的余弦距离最近：{}".format(i,index[1],dis[index[1]]))
            #     print("第{}类与第{}类的余弦距离最远：{}".format(i,index[9],dis[index[9]]))
            #
            #     closest_sim[i] = 1 + F.cosine_similarity(torch.unsqueeze(param[i],0),torch.unsqueeze(param[index[1]],0))
            # print(closest_sim)

    print("==========Epoch:{}===========".format(epoch))
    adjust_learning_rate(classifier_optimizer,epoch)
    # adjust_learning_rate(representation_optimizer,epoch)
    train(args, net, device, train_loader, classifier_optimizer, epoch)

    # classifier层weight长度正则
    model_dict = net.state_dict()
    for name,param in net.named_parameters():
        if "linear" in name and "bias" not in name:
            # update weight
            cls_weight = param.clone().detach()
            for i in range(cls_weight.size()[0]):
                cls_weight[i] /= torch.norm(cls_weight[i])
            state_dict = {name:cls_weight}
            model_dict.update(state_dict)
            net.load_state_dict(model_dict)

    test(epoch)
    writer.close()
