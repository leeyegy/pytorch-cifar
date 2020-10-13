import sys
sys.path.append("../")

from models import  *
import torch
import numpy as np
import os

def analyze_layer_param(model_A,model_B,reduction="mean",norm="L1"):
    model_A_param = {}
    for name,parameters in model_A.named_parameters():
        # print(name,":",parameters.size())
        model_A_param[name]=parameters.cpu().detach().numpy()
    model_diff_param = {}
    for name,parameters in model_B.named_parameters():
        # print(name,":",parameters.size())
        model_diff_param[name]=parameters.cpu().detach().numpy() - model_A_param[name]

    if norm=="L1":
        if reduction == "mean":
            for key in model_diff_param.keys():
                model_diff_param[key] = torch.abs(torch.from_numpy(model_diff_param[key])).mean()
        elif reduction == "sum":
            for key in model_diff_param.keys():
                model_diff_param[key] = torch.abs(torch.from_numpy(model_diff_param[key])).sum()
    else:
        pass
    return model_diff_param

if __name__ == "__main__":
    naturally_trained_res18 = ResNet18().cuda()
    print(naturally_trained_res18)
    naturally_trained_res18 = torch.nn.DataParallel(naturally_trained_res18)
    naturally_trained_res18.load_state_dict(torch.load(os.path.join("../checkpoint","decouple_Decouple18","beta_6.0","class.pth"))['net'])

    Madry_trained_res18 = ResNet18().cuda()
    Madry_trained_res18 = torch.nn.DataParallel(Madry_trained_res18)
    Madry_trained_res18.load_state_dict(torch.load(os.path.join("../checkpoint","decouple_Decouple18","beta_6.0","rep.pth"))['net'])

    param_diff = analyze_layer_param(naturally_trained_res18,Madry_trained_res18,reduction="sum")
    for key in param_diff.keys():
        print(key,":",param_diff[key].data)
