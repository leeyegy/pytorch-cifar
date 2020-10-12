import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)

class MSEAttack():
    def __init__(self,model,eps,perturb_steps,step_size):
        super(MSEAttack,self).__init__()
        self.model = model
        self.epsilon = eps
        self.perturb_steps = perturb_steps
        self.step_size = step_size
        self.loss = sum_squared_error()

    def perturb(self,x_natural,y):
        self.model.eval()
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

        for _ in range(self.perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                adv_feature = self.model(x_adv)
                cln_feature = self.model(x_natural)
                loss = self.loss(adv_feature,cln_feature)
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        return x_adv.clone().detach()

class TreeAttack():
    def __init__(self,main_model,attached_model,eps,perturb_steps,step_size,beta=0.5):
        super(TreeAttack,self).__init__()
        self.main_model = main_model
        self.attached_model = attached_model
        self.epsilon = eps
        self.perturb_steps = perturb_steps
        self.step_size = step_size
        self.beta = beta

    def perturb(self,x_natural,y):
        self.main_model.eval()
        self.attached_model.eval()

        #  generate mask
        attached_mask = (y < 8) & (y > 1)
        attached_left_mask = ~attached_mask

        # generate data
        attached_x_natural = x_natural[attached_mask]
        attached_y = y[attached_mask]
        attached_left_x_natural = x_natural[attached_left_mask]
        attached_left_y = y[attached_left_mask]

        attached_x_adv = attached_x_natural.detach() + 0.001 * torch.randn(attached_x_natural.shape).cuda().detach()
        attached_left_x_adv = attached_left_x_natural.detach() + 0.001 * torch.randn(attached_left_x_natural.shape).cuda().detach()
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

        # for 2 3 4 5 6 7
        for _ in range(self.perturb_steps):
            attached_x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = (1-self.beta)*F.cross_entropy(self.main_model(attached_x_adv), attached_y) + self.beta*F.cross_entropy(self.attached_model(attached_x_adv),attached_y-2)
            grad = torch.autograd.grad(loss_ce, [attached_x_adv])[0]
            attached_x_adv = attached_x_adv.detach() + self.step_size * torch.sign(grad.detach())
            attached_x_adv = torch.min(torch.max(attached_x_adv, attached_x_natural - self.epsilon), attached_x_natural + self.epsilon)
            attached_x_adv = torch.clamp(attached_x_adv, 0.0, 1.0)
        attached_x_adv = torch.clamp(attached_x_adv, 0.0, 1.0)

        # for 0 1 8 9
        for _ in range(self.perturb_steps):
            attached_left_x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(self.main_model(attached_left_x_adv), attached_left_y)
            grad = torch.autograd.grad(loss_ce, [attached_left_x_adv])[0]
            attached_left_x_adv = attached_left_x_adv.detach() + self.step_size * torch.sign(grad.detach())
            attached_left_x_adv = torch.min(torch.max(attached_left_x_adv, attached_left_x_natural - self.epsilon), attached_left_x_natural + self.epsilon)
            attached_left_x_adv = torch.clamp(attached_left_x_adv, 0.0, 1.0)
        attached_left_x_adv = torch.clamp(attached_left_x_adv, 0.0, 1.0)

        # adv data
        x_adv[attached_mask] = attached_x_adv
        x_adv[attached_left_mask] = attached_left_x_adv
        return x_adv.clone().detach()

class TreeEnsembleAttack():
    def __init__(self,main_model,attached_model,attached_model_left,eps,perturb_steps,step_size,beta=0.5):
        super(TreeEnsembleAttack,self).__init__()
        self.main_model = main_model
        self.attached_model = attached_model
        self.epsilon = eps
        self.perturb_steps = perturb_steps
        self.step_size = step_size
        self.attached_model_left = attached_model_left
        self.beta = beta # to balance

    def perturb(self,x_natural,y):
        # 0-10的都会进行测试
        self.main_model.eval()
        self.attached_model.eval()
        self.attached_model_left.eval()

        #  generate mask
        attached_mask = (y < 8) & (y > 1)
        attached_left_mask = ~attached_mask

        # generate data
        attached_x_natural = x_natural[attached_mask]
        attached_y = y[attached_mask]
        attached_left_x_natural = x_natural[attached_left_mask]
        attached_left_y = y[attached_left_mask]

        attached_x_adv = attached_x_natural.detach() + 0.001 * torch.randn(attached_x_natural.shape).cuda().detach()
        attached_left_x_adv = attached_left_x_natural.detach() + 0.001 * torch.randn(attached_left_x_natural.shape).cuda().detach()
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

        # for 2 3 4 5 6 7
        for _ in range(self.perturb_steps):
            attached_x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = (1-self.beta)*F.cross_entropy(self.main_model(attached_x_adv), attached_y) + self.beta*F.cross_entropy(self.attached_model(attached_x_adv),attached_y-2)
            grad = torch.autograd.grad(loss_ce, [attached_x_adv])[0]
            attached_x_adv = attached_x_adv.detach() + self.step_size * torch.sign(grad.detach())
            attached_x_adv = torch.min(torch.max(attached_x_adv, attached_x_natural - self.epsilon), attached_x_natural + self.epsilon)
            attached_x_adv = torch.clamp(attached_x_adv, 0.0, 1.0)
        attached_x_adv = torch.clamp(attached_x_adv, 0.0, 1.0)

        # for 0 1 8 9
        for _ in range(self.perturb_steps):
            attached_left_x_adv.requires_grad_()
            with torch.enable_grad():
                new_target = attached_left_y.clone().detach()
                mask = attached_left_y>7
                new_target[mask] = new_target[mask] - 6
                loss_ce = (1-self.beta)*F.cross_entropy(self.main_model(attached_left_x_adv), attached_left_y) + self.beta*F.cross_entropy(self.attached_model_left(attached_left_x_adv),new_target)
            grad = torch.autograd.grad(loss_ce, [attached_left_x_adv])[0]
            attached_left_x_adv = attached_left_x_adv.detach() + self.step_size * torch.sign(grad.detach())
            attached_left_x_adv = torch.min(torch.max(attached_left_x_adv, attached_left_x_natural - self.epsilon), attached_left_x_natural + self.epsilon)
            attached_left_x_adv = torch.clamp(attached_left_x_adv, 0.0, 1.0)
        attached_left_x_adv = torch.clamp(attached_left_x_adv, 0.0, 1.0)

        # adv data
        x_adv[attached_mask] = attached_x_adv
        x_adv[attached_left_mask] = attached_left_x_adv
        return x_adv.clone().detach()
