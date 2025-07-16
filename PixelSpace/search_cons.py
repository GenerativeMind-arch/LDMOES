import re  #用于处理正则表达式
import torch
from compute_cost import compute_conv_cost, KERNEL_SIZE , dict2namespace
import yaml
import pickle
import argparse




def find_bench_loss(file):   #这里的file应该是搜索时的file
    losses = []
    with open(file,'r') as f:

         for line in f.readlines():
            if 'ops' in line:

                ops_str = line.split('ops:')[-1].split('loss')[0].strip()
                if not ops_str.endswith(']'):
                    ops_str += ']'  # 手动补全
                ops = eval(ops_str)


                mark = True
                for op in ops:
                    if op[0] !=1 or op[1] != 2:

                        mark = False
                if mark:
                    loss = float(re.findall(r'loss=(\d+\.\d+)', line)[0])
                    losses.append(loss)
    return losses

def search_best_local(args):
    with open(args.cost_file, "rb") as f:
        COST = pickle.load(f)  
    bench_losses=find_bench_loss(args.log_file)
    opt_ops = []
    layer_idx = 0
    shift = 0
    global_cost = 0
    
    with open(f'configs/{args.config}') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    stage_num =  2*len(config.model.ch_mult) + 2

    with open(args.log_file,'r') as f:
        lines = list(f.readlines())
        for t_stage in range(stage_num-1):   #[0-8]
            opt_loss = 1e8
            opt_cost =1e8
            opt_op = []
            for line in lines:

                if 'stage' not in line and 'ops' not in line:
                    continue
                if 'stage' in line :

                    stage =  int(re.findall(r'stage:(\d+)', line)[0])
                    if stage > t_stage:
                        break
                if stage != t_stage:
                    continue
                if 'ops' in line:
                    # ops = line.split('ops:')[-1].split(',')[0].split(' ')
                    # ops = [int(op) for op in ops]
                    ops = eval( line.split('ops:')[-1].split('loss')[0].strip()[:-1] ) # [(0, 0), (0, 0), (0, 0), (0, 0)]
                    loss = float(re.findall(r'loss=(\d+\.\d+)', line)[0])
                    shift = len(ops)

                    if loss <= bench_losses[t_stage]*args.loss_scale:
                        op_cost = 0
                        for i,op in enumerate(ops):
                            op_cost += COST[layer_idx+i][op[0]][op[1]]

                        if op_cost < opt_cost:
                            opt_cost = op_cost
                            opt_loss = loss
                            opt_op = ops
                        elif op_cost == opt_cost:
                            if loss < opt_loss:
                                opt_loss = loss
                                opt_op = ops
            
            global_cost += opt_cost
            layer_idx+= shift
            opt_ops.extend(opt_op)
            print(t_stage)
            print(opt_op)
    
    print(layer_idx)
    print(opt_ops)
    print(f"MACs:{global_cost} MB")
    return opt_ops


def get_ops_losses(file):
    with open(file,'r') as f:
        lines = list(f.readlines())

    stage_ops_loss = {}
    block_layers ={}
    for line in lines:
        if 'stage' in line:
            stage =  int(re.findall(r'stage:(\d+)', line)[0])
        if 'ops' in line:
            ops = line.split('ops:')[-1].split(',')[0].split(' ')
            ops = [int(op) for op in ops]
            loss = float(re.findall(r'rse_loss=(\d+\.\d+)', line)[0])
            if stage not in stage_ops_loss.keys():
                stage_ops_loss[stage] = {}
            stage_ops_loss[stage][tuple(ops)] = loss
            if stage not in block_layers.keys():
                block_layers[stage] = len(ops)
    
    return stage_ops_loss, block_layers


def random_select():
    cost_cons = 4577 #celeba是12893定义成本值大小，范围是上下1%
    with open("cifar10_cost.pkl", "rb") as f:
        COST = pickle.load(f)

    l = len([1, 0, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,2, 0, 0, 0, 0, 0, 0])
    while True:
        curr_cost = 0
        ops = torch.randint(0,3,size=(l,))
        for i,op in enumerate(ops):
            curr_cost+= COST[i][op]
        if curr_cost> cost_cons*0.99 and curr_cost < cost_cons*1.01:
            print('random net',ops.tolist())
            print(f"MACs:{curr_cost} MB")
            break

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, help="evaluation log file of super net")
    parser.add_argument("--config", type=str, help="config path")
    parser.add_argument("--loss_scale", type=float,default=1.0)
    parser.add_argument("--cost_file", type=str,help='pre-computed cost file path')
    args = parser.parse_args()
    search_best_local(args)
    pass

    



        
            
        

