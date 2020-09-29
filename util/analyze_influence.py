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
from math import log
from torch.nn import functional as F


import pytorch_influence_functions as ptif

if __name__ == "__main__":
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
    #
    # # model
    # naturally_trained_res18 = ResNet18().cuda()
    # naturally_trained_res18 = torch.nn.DataParallel(naturally_trained_res18)
    # naturally_trained_res18.load_state_dict(torch.load("../checkpoint/FOCAL_CE_ResNet18/ckpt.pth")['net'])
    # print("========= naturally_trained_res18 ============= ")
    # analyze_easy_hard_class_level(naturally_trained_res18)
    # print("=============================================== ")

    Madry_trained_res18 = ResNet18().cuda()
    Madry_trained_res18 = torch.nn.DataParallel(Madry_trained_res18)
    Madry_trained_res18.load_state_dict(torch.load("../checkpoint/amart_ResNet18/ckpt.pth")['net'])
    # print("========= adv_trained_res18 ============= ")
    # analyze_easy_hard_class_level(Madry_trained_res18)
    # analyze_infomation_entropy_class_level(Madry_trained_res18)
    # print("=============================================== ")

    # Supplied by the user:
    # model = get_my_model()
    # trainloader, testloader = get_my_dataloaders()

    ptif.init_logging()
    config = ptif.get_default_config()
    config['outdir'] = '../data/'
    config['log_filename'] = '../log/'
    config['gpu'] = 0

    influences, harmful, helpful = ptif.calc_img_wise(config, Madry_trained_res18, trainloader, testloader)

    # do someting with influences/harmful/helpful

