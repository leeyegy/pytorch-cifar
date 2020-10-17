__author__ = 'Haohan Wang'

import numpy as np
from scipy import signal
from torchvision import datasets, transforms
import torchvision
import torch
import os
from PIL import Image

def fft(img):
    return np.fft.fft2(img)


def fftshift(img):
    return np.fft.fftshift(fft(img))


def ifft(img):
    return np.fft.ifft2(img)


def ifftshift(img):
    return ifft(np.fft.ifftshift(img))


def distance(i, j, imageSize, r):
    dis = np.sqrt((i - imageSize/2) ** 2 + (j - imageSize/2) ** 2)
    if dis < r:
        return 1.0
    else:
        return 0

def mask_radial(img, r):
    rows, cols = img.shape
    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            mask[i, j] = distance(i, j, imageSize=rows, r=r)
    return mask


def generateSmoothKernel(data, r):
    result = np.zeros_like(data)
    [k1, k2, m, n] = data.shape
    mask = np.zeros([3,3])
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                mask[i,j] = 1
            else:
                mask[i,j] = r
    mask = mask
    for i in range(m):
        for j in range(n):
            result[:,:, i,j] = signal.convolve2d(data[:,:, i,j], mask, boundary='symm', mode='same')
    return result


def generateDataWithDifferentFrequencies_GrayScale(Images, r):
    Images_freq_low = []
    mask = mask_radial(np.zeros([28, 28]), r)
    for i in range(Images.shape[0]):
        fd = fftshift(Images[i, :].reshape([28, 28]))
        fd = fd * mask
        img_low = ifftshift(fd)
        Images_freq_low.append(np.real(img_low).reshape([28 * 28]))

    return np.array(Images_freq_low)

def generateDataWithDifferentFrequencies_3Channel(Images, r):
    Images_freq_low = []
    Images_freq_high = []
    mask = mask_radial(np.zeros([Images.shape[1], Images.shape[2]]), r)
    for i in range(Images.shape[0]):
        tmp = np.zeros([Images.shape[1], Images.shape[2], 3])
        for j in range(3):
            fd = fftshift(Images[i, :, :, j])
            fd = fd * mask
            img_low = ifftshift(fd)
            tmp[:,:,j] = np.real(img_low)
        Images_freq_low.append(tmp)
        tmp = np.zeros([Images.shape[1], Images.shape[2], 3])
        for j in range(3):
            fd = fftshift(Images[i, :, :, j])
            fd = fd * (1 - mask)
            img_high = ifftshift(fd)
            tmp[:,:,j] = np.real(img_high)
        Images_freq_high.append(tmp)

    return np.array(Images_freq_low), np.array(Images_freq_high)

if __name__ == '__main__':
    import sys
    version = sys.version_info
    import pickle

    # Data
    print('==> Preparing data..')
    # setup data loader
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10(root='/home/Leeyegy/.torch/datasets/', train=True, download=True,
                                            transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=10)
    testset = torchvision.datasets.CIFAR10(root='/home/Leeyegy/.torch/datasets/', train=False, download=True,
                                           transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=10)

    train_images = np.zeros((50000, 32, 32, 3), dtype='uint8')
    train_labels = np.zeros(50000, dtype='int32')
    # train_loader trans
    for batch_idx, (data, target) in enumerate(train_loader):
        # data trans to [N,H,W,C] and int
        for i in range(data.size()[0]):
            img = np.asarray(transforms.ToPILImage()(data[i]))
            index = batch_idx*128 + i
            train_images[index] = img
            train_labels[index] = target[i]

    test_images = np.zeros((10000, 32, 32, 3), dtype='uint8')
    test_labels = np.zeros(10000, dtype='int32')
    # test_loader trans
    for batch_idx, (data, target) in enumerate(test_loader):
        # data trans to [N,H,W,C] and int
        for i in range(data.size()[0]):
            img = np.asarray(transforms.ToPILImage()(data[i]))
            index = batch_idx*128 + i
            test_images[index] = img
            test_labels[index] = target[i]

    save_path = os.path.join("../data/CIFAR10/train/low_12")

    # save train img
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    train_image_low_4, train_image_high_4 = generateDataWithDifferentFrequencies_3Channel(train_images, 12)
    train_image_low_4 = np.uint8(train_image_low_4)
    class_idx = np.zeros(10)
    for i in range(50000):
        # create class dir
        class_path = os.path.join(save_path,str(train_labels[i]))
        if not os.path.exists(class_path):
            os.makedirs(class_path)
        # get save filename
        file_name = os.path.join(class_path,str(class_idx[train_labels[i]])+".png")

        image = Image.fromarray(train_image_low_4[i])
        image.save(file_name)

        # add
        class_idx[train_labels[i]] += 1

    # save test img
    save_path = os.path.join("../data/CIFAR10/test/low_12")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    test_image_low_4, test_image_high_4 = generateDataWithDifferentFrequencies_3Channel(test_images, 12)
    test_image_low_4 = np.uint8(test_image_low_4)

    class_idx = np.zeros(10)
    for i in range(10000):
        # create class dir
        class_path = os.path.join(save_path,str(test_labels[i]))
        if not os.path.exists(class_path):
            os.makedirs(class_path)
        # get save filename
        file_name = os.path.join(class_path,str(class_idx[test_labels[i]])+".png")
        image = Image.fromarray(test_image_low_4[i])
        image.save(file_name)

        # add
        class_idx[test_labels[i]] += 1