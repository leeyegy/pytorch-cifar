import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import  numpy as np
import math
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
    if loss == "CS":
        output_ = torch.sqrt(torch.sum(output * output, dim=1))
        output = output / output_ # normalization
        score = torch.FloatTensor(output.size()).to(output)
        for k in range(output.size()[0]):
            for i in range(10):
                score[k,i]=(target_map[str(i)].cuda().float()*output[k]).sum()
        pred = score.max(1,keepdim=True)[1]
        return pred.eq(target.view_as(pred)).sum().item()

    else:
        # focal loss and ce loss
        _, predicted = output.max(1)
        return predicted.eq(target).sum().item()

class SPLoss(nn.Module):
    def __init__(self):
        super(SPLoss,self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.N_ratio = 0.8 # {from 0.8 to 1.0}

    def _adjust_N_ratio(self,epoch):
        if epoch<=25:
            self.ratio = 0.7
        elif epoch<=75:
            self.N_ratio = 0.8
        elif epoch <=95:
            self.N_ratio = 0.9
        else:
            self.N_ratio = 1.0

    def _ce_loss(self,output,target):
        log_softmax = nn.LogSoftmax()(output) # [batch_size,10]
        ce_loss = [- log_softmax[:,target[i]].view(log_softmax.size()[0]) for i in range(target.size()[0])] # [batch_size]
        return ce_loss

    def forward(self,output,target,epoch):
        # define ratio
        self._adjust_N_ratio(epoch)

        # calculate CE loss
        ce_loss =self._ce_loss(output,target)

        # justify
        print(ce_loss.size())
        print(ce_loss.mean().data)
        print(ce_loss.sum().data)
        print(self.ce(output,target).data)

        # calculate lambda_t
        ## sort ce_loss
        sorted,index = torch.sort(ce_loss,descending=False)
        lambda_t = sorted[(target.size()[0] * self.N_ratio).int()]

        # calculate q_t
        q_t = 3 * math.tan(math.pi*0.5*(1-(target.size()[0] * self.N_ratio).int()/(target.size()[0]+1))) # original:2
        print(q_t.data)

        # calculate v_t
        v_t = (1-ce_loss/lambda_t) ** (1/(q_t-1))
        print(v_t.size())
        v_t[ce_loss>=lambda_t]=0

        # calculate loss
        loss = ce_loss * v_t
        return loss.mean()


class Margin_Cosine_Similarity_Loss(nn.Module):
    def __init__(self,margin_adv_anchor,margin_adv_most_confusing):
        super(Margin_Cosine_Similarity_Loss,self).__init__()
        self.margin_adv_anchor = margin_adv_anchor
        self.margin_adv_most_confusing = margin_adv_most_confusing
        assert self.margin_adv_most_confusing<1and self.margin_adv_most_confusing>0
        assert self.margin_adv_anchor<1and self.margin_adv_anchor>0

    def forward(self,x1,target):
        """
        :param x1: output of model
        :param target: hard ground truth label
        :return: loss value: >=0 (notice that it can be bigger than one )
        """
        # calculate the distance between adv and anchor
        x2 = torch.DoubleTensor(x1.size()).to(x1)
        for i in range(x2.size()[0]):
            x2[i] = target_map[str(target[i].cpu().numpy())]
        x1_ = torch.sqrt(torch.sum(x1 * x1, dim=1))  # |x1|
        x2_ = torch.sqrt(torch.sum(x2 * x2, dim=1))  # |x2|
        adv_anchor_cos = torch.sum(x1 * x2, dim=1) / (x1_ * x2_)
        adv_anchor_cos = 1.0 - adv_anchor_cos

        # get most confusing label
        x1 = x1 / x1_ # normalization
        score = torch.FloatTensor(x1.size()).to(x1)
        for k in range(x1.size()[0]):
            for i in range(10):
                score[k,i]=(target_map[str(i)].cuda().float()*x1[k]).sum()
        score,_ = torch.sort(score,dim=1,descending=True)
        most_confusing_label = torch.LongTensor(target.size()).to(target)
        for i in range(most_confusing_label.size()[0]):
            most_confusing_label[i] = score[i,0] if score[i,0] != target[i] else score[i,1]

        # calculate the distance between adv and most confusing label
        x2 = torch.DoubleTensor(x1.size()).to(x1)
        for i in range(x2.size()[0]):
            x2[i] = target_map[str(most_confusing_label[i].cpu().numpy())]
        x1_ = torch.sqrt(torch.sum(x1 * x1, dim=1))  # |x1|
        x2_ = torch.sqrt(torch.sum(x2 * x2, dim=1))  # |x2|
        adv_most_confusing_cos = torch.sum(x1 * x2, dim=1) / (x1_ * x2_)
        adv_most_confusing_cos = 1.0 - adv_most_confusing_cos

        # calculate final loss
        penalty_loss = torch.max(self.margin_adv_most_confusing-adv_most_confusing_cos,0) # if adv_most_confusing_cos is bigger than margin, then let it go
        anchor_loss = torch.max(adv_anchor_cos-self.margin_adv_anchor,0) # if adv_anchor_cos is smaller than margin, then let it go;; do not push the dis towards 0
        loss = torch.mean(penalty_loss + anchor_loss)
        return loss

class Focal_Loss(nn.Module):
    def __init__(self,s=64.0,m=0.35,gamma=2,eps=1e-7,mode="cosine"):
        super(Focal_Loss,self).__init__()
        self.s = s
        self.m = m
        self.gamma= gamma
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()
        self.mode = mode
    def forward(self,output,target):
        """
        :param input: output of model
        :param target: hard ground truth label
        :return: loss value
        """
        if self.mode == "cosine":
            phi = output -self.m
            one_hot = torch.zeros(output.size())
            one_hot.scatter_(1,target.view(-1,1).long().cpu(),1)
            one_hot = one_hot.cuda()
            output = (one_hot * phi ) + ((1.0-one_hot)*output)
            output *= self.s
        elif self.mode == "normal":
            # print("normal_mode")
            pass
        logp = self.ce(output,target)
        p = torch.exp(-logp)
        loss = (1-p) ** self.gamma * logp
        return  loss.mean()


class Cosine_Similarity_Loss(nn.Module):
    def __init__(self):
        super(Cosine_Similarity_Loss,self).__init__()
    def forward(self,x1,target):
        """
        :param x1: output of model
        :param target: hard ground truth label
        :return: loss value: >=0 (notice that it can be bigger than one )
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