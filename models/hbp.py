import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .resnet_hbp import *


class HBPNet(nn.Module):
    def __init__(self):
        super(HBPNet, self).__init__()

        ## Modify dimensions based on model

        # resnet 18,34
        self.proj0 = nn.Conv2d(512, 8192, kernel_size=1, stride=1)
        self.proj1 = nn.Conv2d(512, 8192, kernel_size=1, stride=1)
        self.proj2 = nn.Conv2d(512, 8192, kernel_size=1, stride=1)

        # resnet 50, ...
        # self.proj0 = nn.Conv2d(2048, 8192, kernel_size=1, stride=1)
        # self.proj1 = nn.Conv2d(2048, 8192, kernel_size=1, stride=1)
        # self.proj2 = nn.Conv2d(2048, 8192, kernel_size=1, stride=1)

        # fc layer
        self.fc_concat = torch.nn.Linear(8192 * 3, 200)


        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.Linear):
        #         m.weight.data.normal_(0, 0.01)
        #         m.bias.data.zero_()

        self.softmax = nn.LogSoftmax(dim=1)
        self.avgpool = nn.AvgPool2d(kernel_size=14)

        ## select base-model
        # self.features = resnet_model.resnet34(pretrained=True,
        #                                       model_root='/data/guijun/HBP_finegrained/pth/resnet34.pth')
        # self.features = resnet_hbp.resnet18(pretrained=True,
        #                                       model_root='/data/guijun/HBP_finegrained/pth/resnet50.pth')
        self.features = resnet18()

    def forward(self, x):
        batch_size = x.size(0)
        feature4_0, feature4_1, feature4_2 = self.features(x)
        print(0)
        print(feature4_0.size())
        print(feature4_1.size())
        print(feature4_2.size())

        feature4_0 = self.proj0(feature4_0)
        feature4_1 = self.proj1(feature4_1)
        feature4_2 = self.proj2(feature4_2)

        print(1)
        print(feature4_0.size())
        print(feature4_1.size())
        print(feature4_2.size())

        inter1 = feature4_0 * feature4_1
        inter2 = feature4_0 * feature4_2
        inter3 = feature4_1 * feature4_2

        print(2)
        print(inter1.size())
        print(inter2.size())
        print(inter3.size())

        inter1 = self.avgpool(inter1).view(batch_size, -1)
        inter2 = self.avgpool(inter2).view(batch_size, -1)
        inter3 = self.avgpool(inter3).view(batch_size, -1)

        print(3)
        print(inter1.size())
        print(inter2.size())
        print(inter3.size())

        result1 = torch.nn.functional.normalize(torch.sign(inter1) * torch.sqrt(torch.abs(inter1) + 1e-10))
        result2 = torch.nn.functional.normalize(torch.sign(inter2) * torch.sqrt(torch.abs(inter2) + 1e-10))
        result3 = torch.nn.functional.normalize(torch.sign(inter3) * torch.sqrt(torch.abs(inter3) + 1e-10))

        print(4)
        print(inter1.size())
        print(inter2.size())
        print(inter3.size())

        print(5)
        result = torch.cat((result1, result2, result3), 1)
        print(result.size())
        result = self.fc_concat(result)
        print(result.size())
        return self.softmax(result)
