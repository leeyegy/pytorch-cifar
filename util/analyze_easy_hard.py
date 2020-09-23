import sys
sys.path.append("../")

from models import  *
import torch
import numpy as np
import torch.nn as nn
from advertorch.attacks import LinfPGDAttack
import torchvision.transforms as transforms
import torchvision
from advertorch.context import ctx_noparamgrad_and_eval

def _analyze(prediction,ground_trueth,stat_correct,stat_total):
    for i in range(ground_trueth.size()[0]):
        stat_correct[ground_trueth[i]] += (prediction[i] == ground_trueth[i])
        stat_total[ground_trueth[i]] += 1

def analyze_easy_hard_class_level(model):
    model.eval()
    # adversary
    adversary = LinfPGDAttack(model,eps=0.03137,nb_iter=10,eps_iter=0.007,loss_fn=nn.CrossEntropyLoss(),rand_init=True)

    cln_stat_correct = torch.zeros([10]).cuda()
    cln_stat_total = torch.zeros([10]).cuda()
    adv_stat_correct = torch.zeros([10]).cuda()
    adv_stat_total = torch.zeros([10]).cuda()

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            outputs = model(inputs)
        prediction = outputs.max(1,keepdim=True)[1].view_as(targets)
        _analyze(prediction,targets,cln_stat_correct,cln_stat_total)

        with ctx_noparamgrad_and_eval(model):
            pgd_data = adversary.perturb(inputs.clone().detach(), targets)
        with torch.no_grad():
            outputs = model(pgd_data)
        prediction = outputs.max(1,keepdim=True)[1].view_as(targets)
        _analyze(prediction,targets,adv_stat_correct,adv_stat_total)

    print("clean set stat:")
    for i in range(10):
        print("{}/{} ({:.4f})".format(cln_stat_correct[i],cln_stat_total[i],100.0*cln_stat_correct[i]/cln_stat_total[i]))
    print("adv set stat:")
    for i in range(10):
        print("{}/{} ({:.4f})".format(adv_stat_correct[i],adv_stat_total[i],100.0*adv_stat_correct[i]/adv_stat_total[i]))

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
    root='/home/Leeyegy/.torch/datasets', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='/home/Leeyegy/.torch/datasets', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

# model
naturally_trained_res18 = ResNet18().cuda()
naturally_trained_res18 = torch.nn.DataParallel(naturally_trained_res18)
naturally_trained_res18.load_state_dict(torch.load("../checkpoint/CE_ResNet18/ckpt.pth")['net'])
print("========= naturally_trained_res18 ============= ")
analyze_easy_hard_class_level(naturally_trained_res18)
print("=============================================== ")

Madry_trained_res18 = ResNet18().cuda()
Madry_trained_res18 = torch.nn.DataParallel(Madry_trained_res18)
Madry_trained_res18.load_state_dict(torch.load("../checkpoint/CE_ResNet18/adv_PGD_0.03137/ckpt.pth")['net'])
print("========= adv_trained_res18 ============= ")
analyze_easy_hard_class_level(Madry_trained_res18)
print("=============================================== ")

