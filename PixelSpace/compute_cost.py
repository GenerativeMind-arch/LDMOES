import torch
from models.student_supernet import *
import os
import yaml
import argparse
from thop import profile
from thop import clever_format
from models.diffusion import Model
import pickle
import argparse
from runners.model_id import id2ops


KERNEL_SIZE = [1,3,5]
CHANNEL_NUM = [0.5, 0.8, 1.0, 1.2]

def round_ch_num(ch_num):
    new_ch_num = round(ch_num/32) *32
    return new_ch_num

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def compute_whole_net(args, forward_op=None):
    with open(f'configs/{args.config}') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    if forward_op is None:
        num_res_blocks = config.model.num_res_blocks
        _, _, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_resolutions = len(ch_mult)
        resolution = config.data.image_size
        layer_num = num_res_blocks* num_resolutions*2 + 2*2 + (num_res_blocks+1)*num_resolutions*2

        forward_op = [1] * layer_num

    model = StandAloneNet(config, forward_op) 
    model.eval()                  
    input = torch.randn(1, 3, config.data.image_size, config.data.image_size)

    t = torch.randint( low=0, high=1000,size=(1,))
    macs, params = profile(model, inputs=(input,t,))

    macs, params = clever_format([macs, params], "%.3f")
    print(f"MACs:{macs}, Params:{params}")



def compute_conv_cost(args):
    COST = []
    with open(f'configs/{args.config}') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    num_res_blocks = config.model.num_res_blocks
    _, _, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
    num_resolutions = len(ch_mult)
    resolution = config.data.image_size
    # layer_num = num_res_blocks* num_resolutions*2 + 2*2 + (num_res_blocks+1)*num_resolutions*2
    # forward_op = [1] * layer_num
    # model = StandAloneNet(config, forward_op)
    model = Model(config)


    def compute_block(curr_res, net):
        costs = []
        out_ch, in_ch,_,_ =  net.conv1.weight.shape  #net.conv.weight.shape应该是[输出通道数，输入通道数，分辨的长和宽]
        for k_size in KERNEL_SIZE:
            cost = []
            for ch_num in CHANNEL_NUM:
                padding = (k_size - 1) // 2
                out_ch_tmp = round_ch_num(ch_num * out_ch)
                conv = torch.nn.Conv2d(in_ch, out_ch_tmp, kernel_size=k_size, stride=1, padding=padding)
                input = torch.randn(1, conv.weight.shape[1] , curr_res, curr_res)
                macs, params = profile(conv, inputs=(input,))
                # costs.append(macs/1e6)
                cost.append(macs/1e6)
            costs.append(cost)
        COST.append(costs)

        costs = []
        out_ch, in_ch,_,_ =  net.conv2.weight.shape
        for k_size in KERNEL_SIZE:
            cost = []
            for ch_num in CHANNEL_NUM:
                padding = (k_size - 1) // 2
                in_ch_tmp = round_ch_num(ch_num * in_ch)
                conv = torch.nn.Conv2d(in_ch_tmp, out_ch,kernel_size=k_size, stride=1, padding=padding)
                input = torch.randn(1, conv.weight.shape[1] , curr_res, curr_res)
                macs, params = profile(conv, inputs=(input,))
                cost.append(macs/1e6)
            costs.append(cost)
        COST.append(costs)
    

    up_cost = 0
    down_cost = 0
    # down sample
    curr_res = resolution 
    for i_level in range(num_resolutions):
        for i_block in range(num_res_blocks):
            compute_block(curr_res, model.down[i_level].block[i_block])
        if i_level != num_resolutions-1:
            input = torch.randn(1, model.down[i_level].downsample.in_channels , curr_res, curr_res)
            macs, params = profile(model.down[i_level].downsample, inputs=(input,))
            down_cost+= macs/1e6
            curr_res = curr_res // 2

    compute_block(curr_res, model.mid.block_1)
    compute_block(curr_res, model.mid.block_2)

    #up
    for i_level in reversed(range(num_resolutions)):
        for i_block in range(num_res_blocks+1):
            compute_block(curr_res, model.up[i_level].block[i_block])
        if i_level != 0:
            input = torch.randn(1, model.up[i_level].upsample.in_channels , curr_res, curr_res)
            macs, params = profile(model.up[i_level].upsample, inputs=(input,))
            up_cost+= macs/1e6
            curr_res = curr_res * 2

    print('base:',sum([p[1][2] for p in COST ]),'M MACs')  
    print('up:',up_cost,'M MACs')  
    print('down:',down_cost,'M MACs') 

    # print(COST[0])x

    with open(args.cost_file, "wb") as f:
        pickle.dump(COST, f)
    return COST



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config path")
    parser.add_argument("--cost_file", type=str,help='pre-computed cost file path')
    args = parser.parse_args()
    compute_conv_cost(args)
    forward_op = None
    pass