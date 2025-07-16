import re
import pickle



log_file = './final_search_log.log'
cost_file = 'church_cost.pth'
loss_scale = 1.0

def find_bench_loss(file):
    losses = []
    with open(file,'r') as f:
         for line in f.readlines():
            ops = line.split('ops:')[-1].split(',')[0].split(' ')
            ops = [int(op) for op in ops]
            if all(op==1 for op in ops):
                loss = float(re.findall(r'loss:(\d+\.\d+)', line)[0])
                losses.append(loss)
    return losses


def search_best_local():
    with open(cost_file, "rb") as f:
        COST = pickle.load(f)  
    bench_losses=find_bench_loss(log_file)
    opt_ops = []
    layer_idx = 0
    shift = 0
    global_cost = 0
    stage_num = 11
    

    with open(log_file,'r') as f:
        lines = list(f.readlines())
        for t_stage in range(stage_num):
            opt_loss = 1e8
            opt_cost =1e8
            opt_op = []
            for line in lines:
                stage =  int(re.findall(r'stage:(\d+)', line)[0])
                if stage > t_stage:
                    break
                if stage < t_stage:
                    continue

                ops = line.split('ops:')[-1].split(',')[0].split(' ')
                ops = [int(op) for op in ops]
                loss = float(re.findall(r'loss:(\d+\.\d+)', line)[0])
                shift = len(ops)

                if loss <= bench_losses[t_stage]*loss_scale:
                    op_cost = 0
                    for i,op in enumerate(ops):
                        op_cost += COST[layer_idx+i][op]
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
    
    print(opt_ops)
    print(f"MACs:{global_cost} MB")
    return opt_ops


if __name__ == '__main__':
    search_best_local()