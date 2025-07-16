import torch
import os
import yaml
import argparse
from thop import profile
from thop import clever_format
import pickle
import argparse
from ldm.modules.diffusionmodules.Unet_student import AloneUNet
from ldm.modules.diffusionmodules.Unet_teacher import TeacherUNet, ResBlock
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

KERNEL_SIZE = [1,3,5]

# cfg_file= 'configs/latent-diffusion/church_retrain.yaml'
cfg_file= 'configs/latent-diffusion/celebahq_retrain.yaml'


def compute_whole_net(forward_ops=None):

   
    configs = OmegaConf.load(cfg_file)
    student_config= configs.model.params.student_config
    n_levels = len( student_config.params.channel_mult)
    num_res_blocks  = student_config.params.num_res_blocks
    layer_nums = [ (num_res_blocks + 1)*2] * (n_levels-1) + [ num_res_blocks*2] + [2*2] + [ (num_res_blocks + 1+1)*2] * (n_levels-1) + [ (num_res_blocks + 1)*2] 
    total_layers = sum(layer_nums)
    if forward_ops is None:
        forward_ops = [1] * total_layers

    # id 1 base
    # forward_ops = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 2, 1, 1]

    student_config['params']['forward_ops'] = forward_ops
    model = instantiate_from_config(student_config)
    model.eval()         
    
    # configs = OmegaConf.load(cfg_file)
    # unet_config= configs.model.params.unet_config
    # model = instantiate_from_config(unet_config)
    # model.eval()              

    in_ch = student_config.params.in_channels
    input = torch.randn(1, in_ch, student_config.params.image_size,  student_config.params.image_size)
    # input = torch.randn(1, in_ch, unet_config.params.image_size, unet_config.params.image_size)

    t = torch.randint( low=0, high=1000,size=(1,))
    macs, params = profile(model, inputs=(input,t,))
    macs, params = clever_format([macs, params], "%.3f")
    print(f"MACs:{macs}, Params:{params}")



def compute_conv_cost():
    COST = []
    
    configs = OmegaConf.load(cfg_file)

    unet_config= configs.model.params.unet_config
 
    model = instantiate_from_config(unet_config)
    model.eval()  

    def compute_res_block(curr_res, res_block):
        costs = []
        out_ch, in_ch,_,_ =  res_block.in_layers[-1].weight.shape
        for k_size in KERNEL_SIZE:
            padding = (k_size - 1) // 2
            conv = torch.nn.Conv2d(in_ch, out_ch,kernel_size=k_size, stride=1, padding=padding)
            input = torch.randn(1, conv.weight.shape[1] , curr_res, curr_res)
            macs, params = profile(conv, inputs=(input,))
            costs.append(macs/1e6)
        COST.append(costs)

        costs = []
        out_ch, in_ch,_,_ =  res_block.out_layers[-1].weight.shape
        for k_size in KERNEL_SIZE:
            padding = (k_size - 1) // 2
            conv = torch.nn.Conv2d(in_ch, out_ch,kernel_size=k_size, stride=1, padding=padding)
            input = torch.randn(1, conv.weight.shape[1] , curr_res, curr_res)
            macs, params = profile(conv, inputs=(input,))
            costs.append(macs/1e6)

        COST.append(costs)
    
    curr_res =  unet_config.params.image_size

    # down 
    n_levels =  len(unet_config.params.channel_mult)
    i_level = 0

    for module in model.input_blocks:
        for layer in module:
            if isinstance(layer, ResBlock):
                compute_res_block(curr_res, layer)
        if module.get_flag():
            if i_level != n_levels -1:
                curr_res = curr_res//2
            i_level+=1


    # mid
    compute_res_block(curr_res, model.middle_block[0])
    compute_res_block(curr_res, model.middle_block[2])

    i_level = n_levels -1
    #up
    for module in model.output_blocks:
        for layer in module:
            if isinstance(layer, ResBlock):
                compute_res_block(curr_res, layer)
        if module.get_flag():
            if i_level != 0:
                curr_res = curr_res*2
            i_level -= 1


    # print('base:',sum([p[1] for p in COST ]),'M MACs')  

    # with open('celebahq_cost.pth', "wb") as f:
    #     pickle.dump(COST, f)  
    
    # print(COST)



if __name__ == '__main__':

   
    # forward_ops = [1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    forward_ops = None
    # compute_whole_net(forward_ops)
    compute_conv_cost()

   