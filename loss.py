import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import  numpy as np
target_map = {"0":torch.from_numpy(np.asarray([0,0.333,0.333,0.333,0.333,0.333,0.333,0.333,0.333,0.333])),
       "1":torch.from_numpy(np.asarray([1,0.,0.,0.,0.,0.,0.,0.,0.,0.])),
       "2":torch.from_numpy(np.asarray([0,-0.118,0.943,-0.118,-0.118,-0.118,-0.118,-0.118,-0.118,-0.118])),
       "3":torch.from_numpy(np.asarray([0,-0.134,1.780e-16,0.935,-0.134,-0.134,-0.134,-0.134,-0.134,-0.134])),
       "4":torch.from_numpy(np.asarray([0,-0.154,1.999e-16,3.997e-16,0.926,-0.154,-0.154,-0.154,-0.154,-0.154])),
       "5":torch.from_numpy(np.asarray([0,-0.183,1.824e-16,3.953e-16,6.081e-17,0.913,-0.183,-0.183,-0.183,-0.183])),
       "6":torch.from_numpy(np.asarray([0,-0.224,1.738e-16,4.469e-16,7.448e-17,-9.930e-17,0.894,-0.224,-0.224,-0.224])),
       "7":torch.from_numpy(np.asarray([0,-0.289,2.134e-16,4.914e-16,6.410e-17,-1.282e-16,1.282e-16,0.866,-0.289,-0.289])),
       "8":torch.from_numpy(np.asarray([0,-0.408,3.108e-16,7.576e-16,9.712e-17,-1.165e-16,2.331e-16,-3.885e-17,0.816,-0.408])),
       "9":torch.from_numpy(np.asarray([0,-0.707,5.103e-16,1.178e-15,1.963e-16,-1.57e-16,3.140e-16,-3.925e-17,-2.355e-16,0.707]))}


def get_correct_num(output,target,loss):
    """
    :param output: [N,10] | cuda tensor
    :param target: [N,]  | cuda tensor
    :return:
    """
    if loss == "CE":
        _, predicted = output.max(1)
        return predicted.eq(target).sum().item()
    else:
        score = torch.FloatTensor(output.size()).to(output)
        for k in range(output.size()[0]):
            for i in range(10):
                score[k,i]=(target_map[str(i)].cuda().float()*output[k]).sum()
        pred = score.max(1,keepdim=True)[1]
        return pred.eq(target.view_as(pred)).sum().item()

class Cosine_Similarity_Loss(nn.Module):
    def __init__(self):
        super(Cosine_Similarity_Loss,self).__init__()
    def forward(self,x1,target):
        """
        :param x1: output of model
        :param target: hard ground truth label
        :return: loss value
        """
        x2 = torch.DoubleTensor(x1.size()).to(x1)
        for i in range(x2.size()[0]):
            x2[i] = target_map[str(target[i].cpu().numpy())]

        x1_ = torch.sqrt(torch.sum(x1 * x1, dim=1))  # |x1|
        x2_ = torch.sqrt(torch.sum(x2 * x2, dim=1))  # |x2|
        cos_x1_x2 = torch.sum(x1 * x2, dim=1) / (x1_ * x2_)
        ans = torch.mean(1.0 - cos_x1_x2)
        return ans

# output = torch.cosine_similarity(target_map["0"].view([1,10]),target_map["1"].view([1,10]))
# output = Cosine_Similarity_Loss()(target_map["0"].view([1,10]),target_map["1"].view([1,10]))

# print(output)