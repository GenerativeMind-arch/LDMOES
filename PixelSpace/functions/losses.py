import torch
import logging
from models.student_supernet import KERNEL_SIZE
import numpy as np
import random

def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    #output = model(x, t.float(), last_only =True)
    output = model(x, t.float())
    mse_loss = torch.nn.MSELoss()
    return mse_loss(e, output)

        
def distill_loss(teacher, student,
                 x0: torch.Tensor,
                 t: torch.LongTensor,
                 e: torch.Tensor,
                 b: torch.Tensor, start, forward_op = None, train_sub=False, alpha =0, alpha_scale=1):

    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    # block independent
    mse_loss = torch.nn.MSELoss(reduction='mean')
    loss = 0
    if not train_sub:
        with torch.no_grad():
            guides, guide_hs = teacher(x, t.float())
        outputs = student(x, t.float(), guides = guides, guide_hs = guide_hs, 
                        start =start, forward_op = forward_op)

        guide = guides[start]
        loss = mse_loss(guide, outputs[0])

    else:
        inter_loss = 0


        # step 阶梯函数
        if alpha < 1:
            with torch.no_grad():
                guides, guide_hs = teacher(x, t.float())
            outputs = student(x, t.float(), guides = guides, guide_hs = guide_hs)
            for guide, output in zip(guides, outputs):
                # print(guide.shape, output.shape)
                inter_loss +=  mse_loss(guide, output)
            return inter_loss
        else:
            last_out = student(x, t.float(), only_last =True)
            return mse_loss(last_out,e)
            
    return loss

def search_loss(teacher, student,
                 x: torch.Tensor,
                 t: torch.LongTensor, start, forward_op = None, guides = None, guide_hs= None):

    # a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    # x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    # with torch.no_grad():
    #     guides, guide_hs = teacher(x, t.float())
    # block independent
    outputs = student(x, t.float(), guides = guides, guide_hs = guide_hs, 
                      start =start, forward_op = forward_op)

    mse_loss = torch.nn.MSELoss(reduction='mean')
    guide = guides[start]
    
    #relative l2
    loss = mse_loss(guide, outputs[0])
    var = mse_loss(guide,guide.mean())
    relative_loss = loss / var

    # relative L1 loss
    # l1_loss = torch.nn.L1Loss(reduction='mean')
    # var = mse_loss(guide,guide.mean())
    # loss = l1_loss(guide,outputs[0])
    # relative_loss = loss / torch.sqrt(var)
    
    return loss.item(), relative_loss.item()

loss_registry = {
    'simple': noise_estimation_loss,
}
