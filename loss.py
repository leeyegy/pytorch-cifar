import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import  numpy as np
import math
from models import *
import os
from advertorch.context import ctx_noparamgrad_and_eval
from advertorch.attacks import GradientSignAttack,LinfPGDAttack
import random


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

# get mentor nat_probs
# Model
if os.path.exists(os.path.join("checkpoint", 'clean_ce_res18.pth')):
    mentor = ResNet18()
    mentor = mentor.cuda()
    mentor = torch.nn.DataParallel(mentor)
    mentor_checkpoint = torch.load(os.path.join("checkpoint", 'clean_ce_res18.pth'))
    mentor.load_state_dict(mentor_checkpoint['net'])

def advanced_kl_loss(model,
              x_natural,
              y,
              optimizer,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              beta=6.0,
              distance='l_inf'):
    kl = nn.KLDivLoss(reduction='none')
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    logits = model(x_natural)
    logits_adv = model(x_adv)

    adv_probs = F.softmax(logits_adv, dim=1)

    nat_probs = F.softmax(logits, dim=1)
    true_adv_probs = torch.gather(adv_probs, 1, (y.unsqueeze(1)).long()).squeeze()

    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * ((1.0000001 - true_adv_probs) ** beta))
    loss = loss_robust

    return loss

def advanced_trades_mentor_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                gamma = 1.0,
                distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)

    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))

            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)

    # get mentor nat_probs
    mentor_nat_probs = F.softmax(mentor(x_natural), dim=1)
    loss_natural = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    mentor_nat_probs)

    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss

def trades_dbn_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv,mode="adv"), dim=1),
                                       F.softmax(model(x_natural,mode="adv"), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural,mode="cln")
    loss_natural_cln = F.cross_entropy(logits, y)
    loss_natural_adv = F.cross_entropy(model(x_natural,mode="adv"),y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv,mode="adv"), dim=1),
                                                    F.softmax(model(x_natural,mode="adv"), dim=1))
    loss = loss_natural_cln + loss_natural_adv + beta * loss_robust
    return loss

def trades_threshold_loss(model,
                           x_natural,
                           y,
                           optimizer,
                           step_size=0.003,
                           epsilon=0.031,
                           gamma=0.8,
                           perturb_steps=10,
                           beta=6.0,
                           distance='l_inf'):
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    # filter out
    adv_probs = F.softmax(model(x_adv), dim=1)
    true_adv_probs = torch.gather(adv_probs, 1, (y.unsqueeze(1)).long()).squeeze()
    mask = true_adv_probs <= gamma
    x_adv = x_adv[mask]
    x_natural = x_natural[mask]
    y = y[mask]

    model.train()
    batch_size = len(x_natural)
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust

    return loss,batch_size


def weight_tanh_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                s=1.0,
                distance='l_inf'):
    model.eval()

    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_ce = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss_ce, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    # calculate adv loss
    logits_adv = model(x_adv)
    loss_adv = F.cross_entropy(logits_adv, y)
    # print(loss_adv)

    # calculate weight loss
    std_weight = torch.ones([10, 10]).cuda()
    for i in range(10):
        for j in range(10):
            if i != j:
                std_weight[i, j] = (s*s) / -9
            else:
                std_weight[i, i] = s*s
    for name,param in model.named_parameters():
        if "linear" in name and "bias" not in name:
            cls_weight = param
            weight_ = cls_weight @ cls_weight.T
            loss_weight = torch.norm(std_weight - weight_)
    loss = loss_adv + beta * loss_weight
    return loss

def _cosine_simililarity(x, y):
    # x shape: (N, 1, C)
    # y shape: (1, 2N, C)
    # v shape: (N, 2N)
    # x: (2N, 1, C), y: (1, 2N, C)
    v = F.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0), -1)
    return v
#
# def weight_penalization_loss(model,
#                              x_natural,
#                              y,
#                              optimizer,
#                              step_size=0.003,
#                              epsilon=0.031,
#                              perturb_steps=10,
#                              beta=1.0,
#                              distance='l_inf'):
#     model.eval()
#
#     # generate adversarial example
#     x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
#     for _ in range(perturb_steps):
#         x_adv.requires_grad_()
#         with torch.enable_grad():
#             loss_ce = F.cross_entropy(model(x_adv)[0], y)
#         grad = torch.autograd.grad(loss_ce, [x_adv])[0]
#         x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
#         x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
#         x_adv = torch.clamp(x_adv, 0.0, 1.0)
#     model.train()
#
#     x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
#     # zero gradient
#     optimizer.zero_grad()
#
#     # calculate adv loss
#     logits_adv,feature_adv = model(x_adv)
#     logits,feature_cln = model(x_natural)
#
#     loss_adv = F.cross_entropy(logits_adv, y)
#
#     # # calculate feature loss | 分开方向
#     # representations = torch.cat([feature_cln, feature_adv], dim=0)  # (2N, C)
#     # similarity_matrix = _cosine_simililarity(representations, representations)  # (2N, 2N)
#
#
#     # calculate weight loss | 分开方向
#     for name, param in model.named_parameters():
#         if "linear" in name and "bias" not in name:
#             cls_weight = param
#             dis = torch.ones([10]).cuda()
#             closest_sim = torch.ones([10]).cuda()
#             for i in range(cls_weight.size()[0]):
#                 for k in range(cls_weight.size()[0]):
#                     dis[k] = 1 - F.cosine_similarity(torch.unsqueeze(param[i], 0), torch.unsqueeze(param[k], 0))
#                 sorted, index = torch.sort(dis)
#                 closest_sim[i] = 1 + F.cosine_similarity(torch.unsqueeze(param[i], 0),
#                                                          torch.unsqueeze(param[index[1]], 0))
#             loss_weight = torch.mean(closest_sim)
#
#     loss = loss_adv + beta * loss_weight
#     return loss

def weight_penalization_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    model.eval()

    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_ce = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss_ce, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    # calculate adv loss
    logits_adv = model(x_adv)
    loss_adv = F.cross_entropy(logits_adv, y)

    # calculate weight loss | 分开方向
    for name,param in model.named_parameters():
        if "linear" in name and "bias" not in name:
            cls_weight = param
            dis = torch.ones([10]).cuda()
            closest_sim = torch.ones([10]).cuda()
            for i in range(cls_weight.size()[0]):
                for k in range(cls_weight.size()[0]):
                    dis[k] = 1 - F.cosine_similarity(torch.unsqueeze(param[i],0),torch.unsqueeze(param[k],0))
                sorted,index = torch.sort(dis)
                closest_sim[i] = 1 + F.cosine_similarity(torch.unsqueeze(param[i],0),torch.unsqueeze(param[index[1]],0))
            loss_weight = torch.mean(closest_sim)

    loss = loss_adv + beta * loss_weight
    return loss

def weight_penalization_mart_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=5.0,
                distance='l_inf'):
    kl = nn.KLDivLoss(reduction='none')
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    logits = model(x_natural)
    logits_adv = model(x_adv)
    adv_probs = F.softmax(logits_adv, dim=1)
    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])

    loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)
    nat_probs = F.softmax(logits, dim=1)
    true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()
    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))

    # calculate weight loss
    for name,param in model.named_parameters():
        if "linear" in name and "bias" not in name:
            cls_weight = param
            dis = torch.ones([10]).cuda()
            closest_sim = torch.ones([10]).cuda()
            for i in range(cls_weight.size()[0]):
                for k in range(cls_weight.size()[0]):
                    dis[k] = 1 - F.cosine_similarity(torch.unsqueeze(param[i],0),torch.unsqueeze(param[k],0))
                sorted,index = torch.sort(dis)
                closest_sim[i] = 1 + F.cosine_similarity(torch.unsqueeze(param[i],0),torch.unsqueeze(param[index[1]],0))
            loss_weight = torch.mean(closest_sim)

    loss_mart = loss_adv + 5.0 * loss_robust
    loss = loss_mart + float(beta) * loss_weight

    return loss , loss_mart , float(beta) * loss_weight

def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))

            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    # calculate adv loss
    logits_adv = model(x_adv)
    adv_probs = F.softmax(logits_adv, dim=1)
    true_adv_probs = torch.gather(adv_probs, 1, (y.unsqueeze(1)).long()).squeeze()


    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust
    return true_adv_probs,loss

def advanced_trades_whole_loss(model,
              x_natural,
              y,
              optimizer,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              beta=6.0,
              gamma = 1.0,
              distance='l_inf'):
    kl = nn.KLDivLoss(reduction='none')
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    logits = model(x_natural)
    logits_adv = model(x_adv)

    adv_probs = F.softmax(logits_adv, dim=1)

    loss_clean = F.cross_entropy(logits, y)
    nat_probs = F.softmax(logits, dim=1)
    true_adv_probs = torch.gather(adv_probs, 1, (y.unsqueeze(1)).long()).squeeze()

    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1))
    loss = ((loss_clean + float(beta) * loss_robust ) * ((1.0000001 - true_adv_probs) ** gamma)).mean()
    return loss # 在amart_train文件下跑的atrades版本没有对loss_clean也进行加权

# 更改之后的版本，之前的版本对clean部分也会进行加权
def advanced_trades_loss(model,
              x_natural,
              y,
              optimizer,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              beta=6.0,
              gamma = 1.0,
              distance='l_inf'):
    kl = nn.KLDivLoss(reduction='none')
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    logits = model(x_natural)
    logits_adv = model(x_adv)

    adv_probs = F.softmax(logits_adv, dim=1)

    loss_clean = F.cross_entropy(logits, y)
    nat_probs = F.softmax(logits, dim=1)
    true_adv_probs = torch.gather(adv_probs, 1, (y.unsqueeze(1)).long()).squeeze()
    true_cln_probs = torch.gather(nat_probs,1,(y.unsqueeze(1)).long()).squeeze()

    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * ((1.0000001 - true_adv_probs) ** gamma))
    loss = loss_clean + float(beta) * loss_robust
    return loss

def advanced_v3_trades_loss(model,
              x_natural,
              y,
              optimizer,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              beta=6.0,
              gamma = 1.0,
              distance='l_inf'):
    kl = nn.KLDivLoss(reduction='none')
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    logits = model(x_natural)
    logits_adv = model(x_adv)

    adv_probs = F.softmax(logits_adv, dim=1)

    nat_probs = F.softmax(logits, dim=1)
    true_adv_probs = torch.gather(adv_probs, 1, (y.unsqueeze(1)).long()).squeeze()

    loss_clean =(1.0 / batch_size) *  torch.sum(F.cross_entropy(logits_adv, y)* ((1.0000001 - true_adv_probs) ** gamma))
    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * ((1.0000001 - true_adv_probs) ** gamma))
    loss = loss_clean + float(beta) * loss_robust
    return loss

def advanced_v2_trades_loss(model,
              x_natural,
              y,
              optimizer,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              beta=6.0,
              gamma = 1.0,
              distance='l_inf'):
    kl = nn.KLDivLoss(reduction='none')
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    logits = model(x_natural)
    logits_adv = model(x_adv)

    adv_probs = F.softmax(logits_adv, dim=1)

    loss_clean = F.cross_entropy(logits_adv, y)
    nat_probs = F.softmax(logits, dim=1)
    true_adv_probs = torch.gather(adv_probs, 1, (y.unsqueeze(1)).long()).squeeze()

    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * ((1.0000001 - true_adv_probs) ** gamma))
    loss = loss_clean + float(beta) * loss_robust
    return loss

def mart_topk_loss(model,
              x_natural,
              y,
              previous_batch_data,
              previous_batch_target,
              ratio,
              optimizer,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              beta=6.0,
              distance='l_inf'):
    kl = nn.KLDivLoss(reduction='none')
    model.eval()
    batch_size = len(x_natural)

    # generate adversarial example for this batch
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)


    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

# get the max-ratio loss part
    with torch.no_grad():
        adv_logits = model(x_adv)
        true_adv_probs = torch.gather(F.softmax(adv_logits, dim=1), 1, (y.unsqueeze(1)).long()).squeeze()
        sorted, index = torch.sort(true_adv_probs, descending=False)

    # sort
    this_prev_x_natural = x_natural.clone().detach()[index[0:int(batch_size*ratio)]]
    this_prev_x_adv = x_adv.clone().detach()[index[0:int(batch_size*ratio)]]
    this_prev_y = y.clone().detach()[index[0:int(batch_size*ratio)]]

    model.train()

    x_adv = Variable(torch.clamp(this_prev_x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    logits = model(this_prev_x_natural)

    logits_adv = model(x_adv)

    adv_probs = F.softmax(logits_adv, dim=1)

    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]

    new_y = torch.where(tmp1[:, -1] == this_prev_y, tmp1[:, -2], tmp1[:, -1])

    loss_adv = F.cross_entropy(logits_adv, this_prev_y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

    nat_probs = F.softmax(logits, dim=1)

    true_probs = torch.gather(nat_probs, 1, (this_prev_y.unsqueeze(1)).long()).squeeze()

    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
    loss = loss_adv + float(beta) * loss_robust
    return loss,previous_batch_data,previous_batch_target

def mart_ohem_loss(model,
              x_natural,
              y,
              previous_batch_data,
              previous_batch_target,
              ratio,
              optimizer,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              beta=6.0,
              distance='l_inf'):
    kl = nn.KLDivLoss(reduction='none')
    model.eval()
    batch_size = len(x_natural)
    this_prev_x_natural = torch.ones(x_natural.size()).to(x_natural)
    this_prev_x_adv = torch.ones(x_natural.size()).to(x_natural)
    this_prev_y = torch.ones(y.size()).to(y)

    # calculate previous batch loss
    if previous_batch_data is None:
        pass
    else:
        pre_natural = previous_batch_data["cln"]
        pre_adv = previous_batch_data["adv"]
        pre_target = previous_batch_target

    # generate adversarial example for this batch
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)


    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

# get the max-ratio loss part
    with torch.no_grad():
        adv_logits = model(x_adv)
        true_adv_probs = torch.gather(F.softmax(adv_logits, dim=1), 1, (y.unsqueeze(1)).long()).squeeze()
        _, index = torch.sort(true_adv_probs, descending=False)
    if previous_batch_data is not None:
        # sort
        this_prev_x_natural[0:int(batch_size*ratio)] = x_natural.clone().detach()[index[0:int(batch_size*ratio)]]
        this_prev_x_adv[0:int(batch_size*ratio)] = x_adv.clone().detach()[index[0:int(batch_size*ratio)]]
        this_prev_y[0:int(batch_size*ratio)] = y.clone().detach()[index[0:int(batch_size*ratio)]]
        this_prev_x_natural[int(batch_size * ratio):batch_size] = pre_natural.clone().detach()[0:batch_size-int(batch_size * ratio)]
        this_prev_x_adv[int(batch_size * ratio):batch_size] = pre_adv.clone().detach()[0:batch_size-int(batch_size * ratio)]
        this_prev_y[int(batch_size * ratio):batch_size] = pre_target.clone().detach()[0:batch_size-int(batch_size * ratio)]
    else:
        this_prev_x_natural = x_natural
        this_prev_x_adv = x_adv
        this_prev_y = y

    # update prev
    previous_batch_data = {"cln":x_natural.clone().detach()[index[0:batch_size-int(batch_size*ratio)]],
                           "adv":x_adv.clone().detach()[index[0:batch_size-int(batch_size*ratio)]]}
    previous_batch_target = y.clone().detach()[index[0:batch_size-int(batch_size*ratio)]]


    model.train()

    x_adv = Variable(torch.clamp(this_prev_x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    logits = model(this_prev_x_natural)

    logits_adv = model(x_adv)

    adv_probs = F.softmax(logits_adv, dim=1)

    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]

    new_y = torch.where(tmp1[:, -1] == this_prev_y, tmp1[:, -2], tmp1[:, -1])

    loss_adv = F.cross_entropy(logits_adv, this_prev_y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

    nat_probs = F.softmax(logits, dim=1)

    true_probs = torch.gather(nat_probs, 1, (this_prev_y.unsqueeze(1)).long()).squeeze()

    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
    loss = loss_adv + float(beta) * loss_robust


    return loss,previous_batch_data,previous_batch_target

def non_targeted_attack(model,x_natural,y,step_size=0.007,epsilon=0.031,perturb_steps=10):
    model.eval()
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_ce = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss_ce, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv.clone().detach()

def targeted_attack(model,x_natural,targeted_label,step_size=0.007,epsilon=0.031,perturb_steps=10):
    model.eval()
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    y = torch.ones(x_natural.size()[0])
    y[:] = targeted_label
    y = y.long().cuda()

    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_ce = - F.cross_entropy(model(x_adv), y) # 让loss反向，也就是最后让产生的对抗样本靠近 目标类别 ；；需要进行debug检验
        grad = torch.autograd.grad(loss_ce, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv.clone().detach()

def _ce_loss(output,target):
    log_softmax = nn.LogSoftmax()(output) # [batch_size,10]
    ce_loss = torch.ones(log_softmax.size()[0]).cuda()
    for i in range(target.size()[0]):
        ce_loss[i] = - log_softmax[i, target[i]]
    return ce_loss

def representation_loss(model,x_natural,x_adv,y):
    batch_size = len(x_natural)
    # cosine_dis<x_adv,x_cln> + cosine_dis<x_adv,another_x_adv>
    cln_feature,_ = model(x_natural)
    adv_feature,_ = model(x_adv)
    loss = torch.ones(x_natural.size()[0]).to(x_natural)
    cosine_sim_self = F.cosine_similarity(cln_feature, adv_feature)

    # random sample a adv_img whose target is different
    for i in range(batch_size):
        # random sample
        sample_idx = random.randint(0,batch_size-1)
        while sample_idx == i or y[sample_idx] == y[i]:
            sample_idx = random.randint(0, batch_size-1)
        cosine_sim_others = F.cosine_similarity(torch.unsqueeze(adv_feature[i],0),torch.unsqueeze(adv_feature[sample_idx],0))
        loss[i] = cosine_sim_others - cosine_sim_self[i] + 2
    return loss.mean()

def _madry_loss(model,x_natural,x_adv,y,beta):
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    _,logits_adv = model(x_adv)
    loss_adv = F.cross_entropy(logits_adv, y)
    return loss_adv


def _mart_loss(model,x_natural,x_adv,y,beta):
    kl = nn.KLDivLoss(reduction='none')
    batch_size = len(x_natural)

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    _,logits = model(x_natural)
    _,logits_adv = model(x_adv)

    adv_probs = F.softmax(logits_adv, dim=1)
    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])

    loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)
    nat_probs = F.softmax(logits, dim=1)
    true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()

    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
    loss = loss_adv + float(beta) * loss_robust

    return loss

def ensemble_mart_loss(model,
              x_natural,
              y,
              optimizer,
              epoch,
              emphasize_label=3,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              beta=6.0,
              distance='l_inf'):
    model.eval()
    batch_size = len(x_natural)

    # define adversary
    PGD_nontargted = LinfPGDAttack(model, eps=epsilon, nb_iter=perturb_steps, eps_iter=step_size,
                                  loss_fn=nn.CrossEntropyLoss(), rand_init=True,targeted=False)

    # generate adversarial_example
    with ctx_noparamgrad_and_eval(model):
        adv_data = PGD_nontargted.perturb(x_natural, y)

    model.train()
    optimizer.zero_grad()

    ce_loss = _ce_loss(model(adv_data), y)
    # easy mode
    if (epoch+1) % 5 != 0:
        ce_loss[y != emphasize_label] = 0
    else:
        pass
    return ce_loss.mean()

    # # normal part
    # normal_output = model(adv_normal_data)
    # normal_adv_probs = F.softmax(normal_output, dim=1)
    #
    # loss_normal = F.nll_loss(torch.log(1.0001 - normal_adv_probs + 1e-12), normal_targeted_label) # away from normal_targeted_label
    #
    #
    # # emphasize part
    # emphasize_loss = F.cross_entropy(model(adv_emphasize_data),emphasize_target)
    #
    # loss = (loss_normal*adv_normal_data.size()[0]  + emphasize_loss*adv_emphasize_data.size()[0])/batch_size
    #
    # return loss

# def ensemble_mart_loss(model,
#               x_natural,
#               y,
#               optimizer,
#               emphasize_label=3,
#               step_size=0.007,
#               epsilon=0.031,
#               perturb_steps=10,
#               beta=6.0,
#               distance='l_inf'):
#     model.eval()
#     batch_size = len(x_natural)
#
#     # divide data
#     emphasize_data = x_natural.clone().detach()[y==emphasize_label]
#     normal_data = x_natural.clone().detach()[y!=emphasize_label]
#     emphasize_target = y.clone().detach()[y==emphasize_label]
#     normal_target = y.clone().detach()[y!=emphasize_label]
#     normal_targeted_label = normal_target.clone().detach()
#     normal_targeted_label[:] = emphasize_label
#
#     # define adversary
#     PGD_nontargted = LinfPGDAttack(model, eps=epsilon, nb_iter=perturb_steps, eps_iter=step_size,
#                                   loss_fn=nn.CrossEntropyLoss(), rand_init=True,targeted=False)
#
#     # generate adversarial_example
#     with ctx_noparamgrad_and_eval(model):
#         adv_emphasize_data = PGD_nontargted.perturb(emphasize_data, emphasize_target) #对重要类别进行无目标攻击
#         adv_normal_data = PGD_nontargted.perturb(normal_data, normal_target) #对重要类别进行无目标攻击
#
#     model.train()
#     optimizer.zero_grad()
#
#     # normal part
#     normal_output = model(adv_normal_data)
#     normal_loss = _ce_loss(normal_output,normal_target)
#     emphasize_softmax = F.softmax(normal_output,dim=1)
#     emphasize_softmax = emphasize_softmax[:,emphasize_label]
#
#     # emphasize part
#     emphasize_loss = _ce_loss(model(adv_emphasize_data),emphasize_target)
#
#     loss = (beta*emphasize_loss.sum() + normal_loss.sum())/batch_size + beta * emphasize_softmax.mean()
#
#     return loss

#
# def mse_loss(model,
#               x_natural,
#               y,
#               optimizer,
#               step_size=0.007,
#               epsilon=0.031,
#               perturb_steps=10,
#               beta=6.0,
#               distance='l_inf'):
#     kl = nn.KLDivLoss(reduction='none')
#     model.eval()
#     batch_size = len(x_natural)
#     # generate adversarial example
#     x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
#
#     for _ in range(perturb_steps):
#         x_adv.requires_grad_()
#         with torch.enable_grad():
#             loss_ce = F.cross_entropy(model(x_adv), y)
#         grad = torch.autograd.grad(loss_ce, [x_adv])[0]
#         x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
#         x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
#         x_adv = torch.clamp(x_adv, 0.0, 1.0)
#     model.train()
#
#     x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
#     # zero gradient
#     optimizer.zero_grad()
#
#     logits = model(x_natural)
#
#     logits_adv = model(x_adv)
#
#     adv_probs = F.softmax(logits_adv, dim=1)
#
#     tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
#
#     new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
#
#     loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)
#
#     nat_probs = F.softmax(logits, dim=1)
#
#     true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()
#
#     loss_robust = (1.0 / batch_size) * torch.sum(
#         torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
#     loss = loss_adv + float(beta) * loss_robust
#
#     return loss

def rbs_transformation(x, splited_block):

    clip_num = splited_block - 1
    _, w, h, c = x.shape
    result = np.zeros_like(x, dtype=np.float32)
    clip_index = list(range(clip_num+1))

    clip_width_points = np.random.randint(low=0, high=w, size=[clip_num], dtype=np.int32).tolist()
    clip_width_points.append(w)
    clip_width_points.sort()

    clip_high_points = np.random.randint(low=0, high=h, size=[clip_num], dtype=np.int32).tolist()
    clip_high_points.append(h)
    clip_high_points.sort()

    random.shuffle(clip_index)

    now_index = 0
    for ind, index in enumerate(clip_index):
        if index == 0:
            value = clip_width_points[index] + now_index
            result[:, now_index:value, :, :] = x[:, 0:clip_width_points[index], :, :]
            now_index = value
        else:
            value = clip_width_points[index] - clip_width_points[index-1] + now_index
            result[:, now_index:value, :, :] = x[:, clip_width_points[index-1]:clip_width_points[index], :, :]
            now_index = value

    random.shuffle(clip_index)
    x = np.copy(result)

    now_index = 0
    for ind, index in enumerate(clip_index):
        if index == 0:
            value = clip_high_points[index] + now_index
            result[:, :, now_index:value, :] = x[:, :, 0:clip_high_points[index],:]
            now_index = value
        else:
            value = clip_high_points[index] - clip_high_points[index - 1] + now_index
            result[:, :, now_index:value, :] = x[:, :, clip_high_points[index - 1]:clip_high_points[index], :]
            now_index = value

    return result

def mart_aux_loss(model,
              x_natural,
              y,
              optimizer,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              beta=6.0,
              gamma = 1.0,
              distance='l_inf'):
    kl = nn.KLDivLoss(reduction='none')
    model.eval()


    # generate adversarial example
    # for main model
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv)[0], y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    # 针对产生的对抗样本，产生对应的切割并且shuffle之后的样本，定义label_swap；并且将两部分数据合并之后再次shuffle
    x_adv_shuffled = x_adv.clone().detach()
    x_adv_shuffled = torch.from_numpy(np.transpose(rbs_transformation(np.transpose(x_adv_shuffled.cpu().numpy(),(0,2,3,1)), 4),(0,3,1,2))).cuda()
    # # debug 检验shuffle之后的图片的效果
    # import cv2
    # cv2.imwrite("test.png", np.transpose(x_adv_shuffled[0].cpu().numpy(), [1, 2, 0]) * 255)
    # return

    label_swap_ori = y.clone().detach()
    label_swap_shuffled = y.clone().detach()+10
    # 合并数据
    x_adv_all = torch.ones([x_natural.size()[0]*2,3,32,32]).to(x_adv)
    x_adv_all[0:x_adv.size()[0]] = x_adv
    x_adv_all[x_adv.size()[0]:x_natural.size()[0]*2]= x_adv_shuffled
    y_all = y.repeat(2)
    y_swap_all = label_swap_ori.repeat(2)
    y_swap_all[x_adv.size()[0]:x_natural.size()[0]*2] = label_swap_shuffled
    # 生成随机排列
    mask = np.random.permutation(x_natural.size()[0]*2)

    x_adv_all = x_adv_all[mask]
    y_all = y_all[mask]
    y_swap_all = y_swap_all[mask]
    x_natural = x_natural.repeat(2,1,1,1)
    x_natural = x_natural[mask]

    model.train()

    # 计算总loss
    x_adv = Variable(torch.clamp(x_adv_all, 0.0, 1.0), requires_grad=False)

    # zero gradient
    optimizer.zero_grad()
    batch_size = len(x_natural)

    # for main model
    logits = model(x_natural)[0]
    logits_adv = model(x_adv)[0]

    adv_probs = F.softmax(logits_adv, dim=1)
    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == y_all, tmp1[:, -2], tmp1[:, -1])

    loss_adv = F.cross_entropy(logits_adv, y_all) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)
    nat_probs = F.softmax(logits, dim=1)
    true_probs = torch.gather(nat_probs, 1, (y_all.unsqueeze(1)).long()).squeeze()

    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
    loss_main = loss_adv + float(beta) * loss_robust

    # for aux model
    aux_output = model(x_adv)[1]
    aux_ce_loss = F.cross_entropy(aux_output,y_swap_all)

    loss = loss_main + aux_ce_loss * gamma
    return loss

# def mart_aux_loss(model,
#               x_natural,
#               y,
#               optimizer,
#               step_size=0.007,
#               epsilon=0.031,
#               perturb_steps=10,
#               beta=6.0,
#               distance='l_inf'):
#     kl = nn.KLDivLoss(reduction='none')
#     model.eval()
#     batch_size = len(x_natural)
#
#     # generate aux data
#
#     # # filter out
#     mask = (y<6) & (y>1)
#     x_natural_aux =  x_natural[mask].clone().detach()
#     y_aux = y[mask].clone().detach() - 2
#
#     # generate adversarial example
#     # for main model
#     x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
#     if distance == 'l_inf':
#         for _ in range(perturb_steps):
#             x_adv.requires_grad_()
#             with torch.enable_grad():
#                 loss_ce = F.cross_entropy(model(x_adv,mode="main"), y)
#             grad = torch.autograd.grad(loss_ce, [x_adv])[0]
#             x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
#             x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
#             x_adv = torch.clamp(x_adv, 0.0, 1.0)
#     else:
#         x_adv = torch.clamp(x_adv, 0.0, 1.0)
#
#     # for aux model
#     x_adv_aux = x_natural_aux.detach() + 0.001 * torch.randn(x_natural_aux.shape).cuda().detach()
#
#     if distance == 'l_inf':
#         for _ in range(perturb_steps):
#             x_adv_aux.requires_grad_()
#             with torch.enable_grad():
#                 loss_ce = F.cross_entropy(model(x_adv_aux,mode="aux"), y_aux)
#             grad = torch.autograd.grad(loss_ce, [x_adv_aux])[0]
#             x_adv_aux = x_adv_aux.detach() + step_size * torch.sign(grad.detach())
#             x_adv_aux = torch.min(torch.max(x_adv_aux, x_natural_aux - epsilon), x_natural_aux + epsilon)
#             x_adv_aux = torch.clamp(x_adv_aux, 0.0, 1.0)
#     else:
#         x_adv_aux = torch.clamp(x_adv_aux, 0.0, 1.0)
#
#     model.train()
#
#     x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
#     x_adv_aux = Variable(torch.clamp(x_adv_aux, 0.0, 1.0), requires_grad=False)
#
#     # zero gradient
#     optimizer.zero_grad()
#
#     # for main model
#     logits = model(x_natural,mode="main")
#     logits_adv = model(x_adv,mode="main")
#
#     adv_probs = F.softmax(logits_adv, dim=1)
#     tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
#     new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
#
#     loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)
#     nat_probs = F.softmax(logits, dim=1)
#     true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()
#
#     loss_robust = (1.0 / batch_size) * torch.sum(
#         torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
#     loss_main = loss_adv + float(beta) * loss_robust
#
#     # for aux model
#     batch_size_aux = x_natural_aux.size()[0]
#     logits_aux = model(x_natural_aux,mode="aux")
#     logits_adv_aux = model(x_adv_aux,mode="aux")
#
#     adv_probs_aux = F.softmax(logits_adv_aux, dim=1)
#     tmp1_aux = torch.argsort(adv_probs_aux, dim=1)[:, -2:]
#     new_y_aux = torch.where(tmp1_aux[:, -1] == y_aux, tmp1_aux[:, -2], tmp1_aux[:, -1])
#
#     loss_adv_aux = F.cross_entropy(logits_adv_aux, y_aux) + F.nll_loss(torch.log(1.0001 - adv_probs_aux + 1e-12), new_y_aux)
#     nat_probs_aux = F.softmax(logits_aux, dim=1)
#     true_probs_aux = torch.gather(nat_probs_aux, 1, (y_aux.unsqueeze(1)).long()).squeeze()
#
#     loss_robust_aux = (1.0 / batch_size_aux) * torch.sum(
#         torch.sum(kl(torch.log(adv_probs_aux + 1e-12), nat_probs_aux), dim=1) * (1.0000001 - true_probs_aux))
#     loss_aux = loss_adv_aux + float(beta) * loss_robust_aux
#
#     loss = loss_main + loss_aux
#     return loss

def mart_loss(model,
              x_natural,
              y,
              optimizer,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              beta=6.0,
              distance='l_inf'):
    kl = nn.KLDivLoss(reduction='none')
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    logits = model(x_natural)

    logits_adv = model(x_adv)

    adv_probs = F.softmax(logits_adv, dim=1)

    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])

    loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

    nat_probs = F.softmax(logits, dim=1)

    true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()

    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
    loss = loss_adv + float(beta) * loss_robust

    return loss

def advanced_mart_whole_loss_v2(model,
              x_natural,
              y,
              optimizer,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              beta=6.0,
              gamma=1.0,
              distance='l_inf'):
    kl = nn.KLDivLoss(reduction='none')
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    logits = model(x_natural)

    logits_adv = model(x_adv)

    adv_probs = F.softmax(logits_adv, dim=1)

    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]

    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])

    loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

    nat_probs = F.softmax(logits, dim=1)

    true_adv_probs = torch.gather(adv_probs, 1, (y.unsqueeze(1)).long()).squeeze()

    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1))
    loss = ((loss_adv + float(beta) * loss_robust) * ((1.0000001-true_adv_probs) ** gamma)).mean()

    return loss

def advanced_mart_whole_loss(model,
              x_natural,
              y,
              optimizer,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              beta=6.0,
              gamma=1.0,
              distance='l_inf'):
    kl = nn.KLDivLoss(reduction='none')
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    logits = model(x_natural)

    logits_adv = model(x_adv)

    adv_probs = F.softmax(logits_adv, dim=1)

    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]

    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])

    loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

    nat_probs = F.softmax(logits, dim=1)

    true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()
    true_adv_probs = torch.gather(adv_probs, 1, (y.unsqueeze(1)).long()).squeeze()

    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
    loss = ((loss_adv + float(beta) * loss_robust) * ((1.0000001-true_adv_probs) ** gamma)).mean()

    return loss

def advanced_mart_inverse_loss(model,
              x_natural,
              y,
              optimizer,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              beta=6.0,
              gamma=1.0,
              distance='l_inf'):
    kl = nn.KLDivLoss(reduction='none')
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    logits = model(x_natural)
    logits_adv = model(x_adv)

    adv_probs = F.softmax(logits_adv, dim=1)
    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1]) # misclassified max prob label

    loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)
    nat_probs = F.softmax(logits, dim=1)
    true_adv_probs = torch.gather(adv_probs, 1, (y.unsqueeze(1)).long()).squeeze()

    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * ((2.718281828459-torch.exp(1.0000001 - true_adv_probs))/1.718281828459))
    loss = loss_adv + float(beta) * loss_robust

    return loss

def advanced_mart_mentor_loss(model,
              x_natural,
              y,
              optimizer,
              mentor_model=None,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              beta=6.0,
              gamma=1.0,
              distance='l_inf'):
    kl = nn.KLDivLoss(reduction='none')
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    logits_adv = model(x_adv)

    adv_probs = F.softmax(logits_adv, dim=1)
    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1]) # misclassified max prob label

    loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)
    true_adv_probs = torch.gather(adv_probs, 1, (y.unsqueeze(1)).long()).squeeze()

    # get mentor nat_probs
    mentor_nat_probs = F.softmax(mentor(x_natural), dim=1)


    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), mentor_nat_probs), dim=1) * ((1.0000001 - true_adv_probs) ** gamma))
    loss = loss_adv + float(beta) * loss_robust

    return loss


def advanced_mart_threshold_loss(model,
              x_natural,
              y,
              optimizer,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              beta=6.0,
              gamma=0.8,
              distance='l_inf'):
    kl = nn.KLDivLoss(reduction='none')
    model.eval()
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    # filter out
    adv_probs = F.softmax(model(x_adv), dim=1)
    true_adv_probs = torch.gather(adv_probs, 1, (y.unsqueeze(1)).long()).squeeze()
    mask = true_adv_probs <= gamma
    x_adv = x_adv[mask]
    x_natural = x_natural[mask]
    y = y[mask]

    model.train()
    batch_size = len(x_natural)
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    logits = model(x_natural)
    logits_adv = model(x_adv)

    adv_probs = F.softmax(logits_adv, dim=1)
    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1]) # misclassified max prob label

    loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)
    nat_probs = F.softmax(logits, dim=1)
    true_adv_probs = torch.gather(adv_probs, 1, (y.unsqueeze(1)).long()).squeeze()

    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * ((1.0000001 - true_adv_probs) ))
    loss = loss_adv + float(beta) * loss_robust

    return loss,batch_size


def advanced_mart_loss(model,
              x_natural,
              y,
              optimizer,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              beta=6.0,
              gamma=1.0,
              distance='l_inf'):
    kl = nn.KLDivLoss(reduction='none')
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    logits = model(x_natural)
    logits_adv = model(x_adv)

    adv_probs = F.softmax(logits_adv, dim=1)
    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1]) # misclassified max prob label

    loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)
    nat_probs = F.softmax(logits, dim=1)
    true_adv_probs = torch.gather(adv_probs, 1, (y.unsqueeze(1)).long()).squeeze()

    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * ((1.0000001 - true_adv_probs) ** gamma))
    loss = loss_adv + float(beta) * loss_robust

    return loss

def get_correct_num(output,target,loss):
    """
    :param output: [N,10] | cuda tensor
    :param target: [N,]  | cuda tensor
    :return:
    """
    if loss == "CS":
        output_ = torch.sqrt(torch.sum(output * output, dim=1))
        score = torch.FloatTensor(output.size()).to(output)
        for k in range(output.size()[0]):
            for i in range(10):
                score[k,i]=torch.sum(target_map[str(i)].cuda().float()*output[k]) / (1*output_[i])
        pred = score.max(1,keepdim=True)[1]
        return pred.eq(target.view_as(pred)).sum().item()

    else:
        # focal loss and ce loss
        _, predicted = output.max(1)
        return predicted.eq(target).sum().item()


class ReviewLoss(nn.Module):
    def __init__(self,review_ratio):
        super(ReviewLoss, self).__init__()
        self.review_ratio = review_ratio

    def _ce_loss(self,output,target):
        log_softmax = nn.LogSoftmax()(output) # [batch_size,10]
        ce_loss = torch.ones(log_softmax.size()[0]).cuda()
        for i in range(target.size()[0]):
            ce_loss[i] = - log_softmax[i, target[i]]
        return ce_loss

    def forward(self,output,target):
        # calculate CE loss
        ce_loss =self._ce_loss(output,target)

        # calculate lambda_t
        ## sort ce_loss
        sorted,index = torch.sort(ce_loss,descending=True)
        lambda_t = sorted[int(target.size()[0] * self.review_ratio)]

        ce_loss[ce_loss<lambda_t] = 0
        return ce_loss.mean()

class SPLoss(nn.Module):
    def __init__(self):
        super(SPLoss,self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.N_ratio = 0.8 # {from 0.8 to 1.0}
        self.q_t = 0

    def _adjust_N_ratio_q_t(self,epoch):
        if epoch<=25:
            self.N_ratio = 0.7
            self.q_t = 2
        elif epoch<=75:
            self.N_ratio = 0.8
            self.q_t = 0
        elif epoch <=95:
            self.N_ratio = 0.9
            self.q_t = 0
        else:
            self.N_ratio = 1.0
            self.q_t = 0

    def _ce_loss(self,output,target):
        log_softmax = nn.LogSoftmax()(output) # [batch_size,10]
        ce_loss = torch.ones(log_softmax.size()[0]).cuda()
        for i in range(target.size()[0]):
            ce_loss[i] = - log_softmax[i, target[i]]
        return ce_loss

    def forward(self,output,target,epoch):
        # define ratio
        self._adjust_N_ratio_q_t(epoch)

        # calculate CE loss
        ce_loss =self._ce_loss(output,target)

        # calculate lambda_t
        ## sort ce_loss
        sorted,index = torch.sort(ce_loss,descending=False)
        lambda_t = sorted[int(target.size()[0] * self.N_ratio)]

        # calculate v_t
        v_t = (1-ce_loss/lambda_t) ** (1/(self.q_t-1))
        v_t[ce_loss>=lambda_t]=0
        v_t[v_t>=10] = 10
        # print(v_t.data)
        # print(torch.max(v_t))
        # print(torch.min(v_t))

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


class Easy2hardLoss(nn.Module):
    def __init__(self,ban_target=3):
        super(Easy2hardLoss,self).__init__()
        self.ban_target = ban_target

    def _ce_loss(self,output,target):
        log_softmax = nn.LogSoftmax()(output) # [batch_size,10]
        ce_loss = torch.ones(log_softmax.size()[0]).cuda()
        for i in range(target.size()[0]):
            ce_loss[i] = - log_softmax[i, target[i]]
        return ce_loss

    def forward(self,output,target,epoch):
        """
        :param input: output of model
        :param target: hard ground truth label
        :return: loss value
        """
        ce_loss = self._ce_loss(output, target)

        if epoch<=80:
            # easy mode
            if epoch % 5 != 0:
                ce_loss[target==self.ban_target] = 0
        else:
            # hard mode
            if epoch % 5 != 0 :
                ce_loss[target!=self.ban_target] = 0
        return  ce_loss.mean()

class Ban_Loss(nn.Module):
    def __init__(self,ban_target=3):
        super(Ban_Loss,self).__init__()
        self.ban_target = ban_target

    def _ce_loss(self,output,target):
        log_softmax = nn.LogSoftmax()(output) # [batch_size,10]
        ce_loss = torch.ones(log_softmax.size()[0]).cuda()
        for i in range(target.size()[0]):
            ce_loss[i] = - log_softmax[i, target[i]]
        return ce_loss

    def forward(self,output,target):
        """
        :param input: output of model
        :param target: hard ground truth label
        :return: loss value
        """

        ce_loss = self._ce_loss(output, target)
        ce_loss[target==3] = 0
        return  ce_loss.mean()

class KingLoss(nn.Module):
    def __init__(self,king=3,beta=1):
        super(KingLoss,self).__init__()
        self.king = king
        self.beta = beta
        self.ce = nn.CrossEntropyLoss()

    def _ce_loss(self,output,target):
        log_softmax = nn.LogSoftmax()(output) # [batch_size,10]
        ce_loss = torch.ones(log_softmax.size()[0]).cuda()
        for i in range(target.size()[0]):
            ce_loss[i] = - log_softmax[i, target[i]]
        return ce_loss


    def forward(self,output,target,epoch):
        """
        :param output: output of model [N,10]
        :param target: hard ground truth label
        :return: loss value
        """
        ce_loss = self._ce_loss(output, target)
        # easy mode
        if epoch % 5 != 0:
            ce_loss[target!=self.king] = 0
        else:
            softmax_output = F.softmax(output,dim=1)
            for i in range(target.size()[0]):
                if target[i] != self.king:
                   ce_loss[i] += softmax_output[i,self.king]
        return  ce_loss.mean()

# only for binary classifier
class BalanceLoss(nn.Module):
    def __init__(self):
        super(BalanceLoss,self).__init__()

    def _ce_loss(self,output,target):
        log_softmax = nn.LogSoftmax()(output) # [batch_size,10]
        ce_loss = torch.ones(log_softmax.size()[0]).cuda()
        for i in range(target.size()[0]):
            ce_loss[i] = - log_softmax[i, target[i]]
        return ce_loss

    def forward(self,output,target):
        """
        :param input: output of model
        :param target: hard ground truth label
        :return: loss value
        """
        ce_loss = self._ce_loss(output, target)
        # print(ce_loss[target==1].data)
        ce_loss[target==1] *= 9
        # print(ce_loss[target==1].data)
        return  ce_loss.mean()

class Focal_Loss(nn.Module):
    def __init__(self,s=64.0,m=0.35,gamma=2,eps=1e-7,mode="cosine",individual=False):
        super(Focal_Loss,self).__init__()
        self.s = s
        self.m = m
        self.gamma= gamma
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()
        self.mode = mode
        self.individual = individual

    def _ce_loss(self,output,target):
        log_softmax = nn.LogSoftmax()(output) # [batch_size,10]
        ce_loss = torch.ones(log_softmax.size()[0]).cuda()
        for i in range(target.size()[0]):
            ce_loss[i] = - log_softmax[i, target[i]]
        return ce_loss

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
        if not self.individual:
            logp = self.ce(output,target)
            p = torch.exp(-logp)
            loss = (1-p) ** self.gamma * logp
        else:
            ce_loss = self._ce_loss(output, target)
            p = torch.exp(-ce_loss)
            loss = (1-p) ** self.gamma * ce_loss
        return  loss.mean()

class RBFLoss(nn.Module):
    def __init__(self,gamma=1.0):
        super(RBFLoss,self).__init__()
        self.gamma = gamma

    def forward(self,x1,target):
        """
        :param x1: output of model
        :param target: hard ground truth label
        :return: loss value: [0,1]
        """
        # map hard-label 'target' into soft-label
        x2 = torch.DoubleTensor(x1.size()).to(x1)
        for i in range(x2.size()[0]):
            x2[i] = target_map[str(target[i].cpu().numpy())]

        x1_ = torch.sqrt(torch.sum(x1 * x1, dim=1)).view(x1.size()[0],1)  # |x1|
        x1_ = x1_.repeat(1,10)
        x1 = x1 /x1_
        # calculate norm-2's square
        norm = torch.sum((x1-x2).pow(2),dim=1)

        #calculate RBF function
        RBF = torch.exp(-self.gamma*norm)

        # calculate loss
        loss = 1 - RBF
        return loss.mean()

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