from PIL import Image
import torch
import time
from tqdm import tqdm
import argparse
import yaml
from models.student_supernet import *
import os 
import pickle



def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def compute_conv_latency(args):
    batch_size = 64
    Latency = []
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
        out_ch, in_ch,_,_ =  net.conv1.weight.shape
        for k_size in KERNEL_SIZE:
            padding = (k_size - 1) // 2
            conv = torch.nn.Conv2d(in_ch, out_ch,kernel_size=k_size, stride=1, padding=padding)
            input = torch.randn(batch_size, conv.weight.shape[1] , curr_res, curr_res)
            latency = compute_layer_latency(conv,input)
            costs.append(latency)
        Latency.append(costs)

        costs = []
        out_ch, in_ch,_,_ =  net.conv2.weight.shape
        for k_size in KERNEL_SIZE:
            padding = (k_size - 1) // 2
            conv = torch.nn.Conv2d(in_ch, out_ch,kernel_size=k_size, stride=1, padding=padding)
            input = torch.randn(batch_size, conv.weight.shape[1] , curr_res, curr_res)
            latency = compute_layer_latency(conv,input)
            costs.append(latency)
        Latency.append(costs)

    atte_latency = 0
    up_latency = 0
    down_latency = 0

    # down sample
    curr_res = resolution 
    for i_level in range(num_resolutions):
        for i_block in range(num_res_blocks):
            compute_block(curr_res, model.down[i_level].block[i_block])
            if len(model.down[i_level].attn) > 0:
                input = torch.randn(batch_size, model.down[i_level].attn[i_block].in_channels , curr_res, curr_res)
                atte_latency+=compute_layer_latency(model.down[i_level].attn[i_block],input)
                
        if i_level != num_resolutions-1:
            input = torch.randn(batch_size, model.down[i_level].downsample.in_channels , curr_res, curr_res)
            down_latency += compute_layer_latency(model.down[i_level].downsample,input)
            curr_res = curr_res // 2

    #middle
    compute_block(curr_res, model.mid.block_1)
    compute_block(curr_res, model.mid.block_2)

    #up
    for i_level in reversed(range(num_resolutions)):
        for i_block in range(num_res_blocks+1):
            compute_block(curr_res, model.up[i_level].block[i_block])
            if len(model.up[i_level].attn) > 0:
                input = torch.randn(batch_size, model.up[i_level].attn[i_block].in_channels , curr_res, curr_res)
                atte_latency+=compute_layer_latency(model.up[i_level].attn[i_block],input)
        if i_level != 0:
            input = torch.randn(batch_size, model.up[i_level].upsample.in_channels , curr_res, curr_res)
            up_latency += compute_layer_latency(model.up[i_level].upsample,input)
            curr_res = curr_res * 2

    input = torch.randn(batch_size, model.conv_out.in_channels , curr_res, curr_res)
    out_latency=compute_layer_latency(model.conv_out,input)

    print('conv base:',sum([p[1] for p in Latency ]),'ms')  
    print(f"attn latency:{atte_latency} ms")
    print(f"up latency:{up_latency} ms")
    print(f"down latency:{down_latency} ms")
    print(f"out latency:{out_latency} ms")


    # with open(args.latency_file, "wb") as f:
    #     pickle.dump(Latency, f) 
    return Latency


def compute_layer_latency(model,input):
    model.eval()
    model = model.cuda()
    input = input.cuda()

    with torch.no_grad():
        latency = 0
        launch_num = 1
        for i in range(launch_num):
            num_runs = 200
            num_warmup_runs = 10
            for i in range(num_runs):
                #warm up
                if i == num_warmup_runs:
                    torch.cuda.synchronize()
                    start_time = time.time()
                model(input)
            torch.cuda.synchronize()
            end_time = time.time()
            total_forward = end_time - start_time
            actual_num_runs = num_runs - num_warmup_runs
            latency += total_forward / actual_num_runs 
        latency = latency/launch_num * 1000
    return latency

def compute_latency_ms_pytorch(args):

    forward_op = [1, 0, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 2, 1, 1, 1, 1, 0, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0]
    if args.base:
        forward_op = [1] * len(forward_op)

    with open(f'configs/{args.config}') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    model = StandAloneNet(config, forward_op) 
    model.eval()       
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    model.eval()
    model = model.cuda()
    batch_size =64
    input = torch.randn(batch_size, 3, config.data.image_size, config.data.image_size).cuda()
    t = torch.randint( low=0, high=1000,size=(batch_size,)).cuda()


    with torch.no_grad():
        latency = 0
        launch_num = 1
        for i in range(launch_num):
            num_runs = 200
            num_warmup_runs = 10
            for i in tqdm(range(num_runs)):
                #warm up
                if i == num_warmup_runs:
                    torch.cuda.synchronize()
                    start_time = time.time()
                model(input,t,sample=True)
            torch.cuda.synchronize()
            end_time = time.time()
            total_forward = end_time - start_time
            actual_num_runs = num_runs - num_warmup_runs
            latency += total_forward / actual_num_runs 
        latency = latency/launch_num * 1000

    print(f"latency:{latency} ms")
    return latency

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config path")
    parser.add_argument("--base", action='store_true')
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--latency_file", type=str, default='')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu

    # compute the whole net
    # compute_latency_ms_pytorch(args)
    #compute layer by layer
    compute_conv_latency(args)
    