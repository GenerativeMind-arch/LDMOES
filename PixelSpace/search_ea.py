import os
import sys
import time
import glob
import numpy as np
import pickle
import torch
import logging
import argparse
import torch
import random


torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True


from torch.autograd import Variable
import collections
import sys
sys.setrecursionlimit(10000)
import argparse

import functools
print = functools.partial(print, flush=True)
choice = lambda x: x[np.random.randint(len(x))] if isinstance(x, tuple) else choice(tuple(x))

from search_cons import find_bench_loss, get_ops_losses


class EvolutionSearcher(object):

    def __init__(self, args):
        self.args = args
        self.max_epochs = args.max_epochs
        self.select_num = args.select_num
        self.population_num = args.population_num
        self.m_prob = args.m_prob
        self.crossover_num = args.crossover_num
        self.mutation_num = args.mutation_num
        self.log_dir = args.log_dir #日志文件
        self.checkpoint_name = os.path.join(self.log_dir, 'checkpoint.pth.tar')
        self.memory = []
        self.vis_dict = {}
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        self.candidates = []
        self.nr_layer = args.nr_layer
        self.nr_state = args.nr_state
        self.loss_limit = sum(find_bench_loss(args.search_file))


        with open("/data/JiaoChunXiao/LDMOES/PixelSpace/cost/cifar_cost.pkl", "rb") as f:
            self.COST = pickle.load(f)  

        self.stage_ops_loss, self.block_layers = get_ops_losses(args.search_file)


    def is_legal(self, cand):

        assert isinstance(cand, tuple) and len(cand) == self.nr_layer
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]

        if 'visited' in info:
            return False

        if 'loss' not in info:
            info['loss'] = self.get_cand_loss(cand)


        if info['loss'] > self.loss_limit:
            return False

        info['cost'] = self.get_cand_cost(cand)

        info['visited'] = True
        return True


    def get_cand_cost(self, cand):
        cost = 0
        for i, op in enumerate(cand):
            cost+= self.COST[i][op]
        return cost


    def get_cand_loss(self, cand):
        loss = 0
        start = 0
        for stage in range(len(self.stage_ops_loss)):
            ops_len = self.block_layers[stage]

            ops = cand[start: start+ops_len]
            loss += self.stage_ops_loss[stage][ops]
            start+= ops_len
        return loss


    def update_top_k(self, candidates, *, k, key, reverse=False):
        assert k in self.keep_top_k
        print('select ......')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]


    def stack_random_cand(self, random_func, *, batchsize=10):
        while True:
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand] = {}
                info = self.vis_dict[cand]
            for cand in cands:
                yield cand

    def get_random(self, num):
        print('random select ........')
        cand_iter = self.stack_random_cand(lambda: tuple(np.random.randint(self.nr_state) for i in range(self.nr_layer)))

        while len(self.candidates) < num:
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            # print('random {}/{}'.format(len(self.candidates), num))
        print('random_num = {}'.format(len(self.candidates)))


    def get_mutation(self, k, mutation_num, m_prob):
        # k = 10
        # mutation_num = 25
        # m_prob = 0.1
        assert k in self.keep_top_k
        print('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10   #250

        def random_func():
            cand = list(choice(self.keep_top_k[k]))
            for i in range(self.nr_layer):

                if np.random.random_sample() < m_prob:
                    cand[i] = np.random.randint(self.nr_state)
            return tuple(cand)

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue

            res.append(cand)
            # print('mutation {}/{}'.format(len(res), mutation_num))

        print('mutation_num = {}'.format(len(res)))
        return res


    def get_crossover(self, k, crossover_num):
        # k = 10
        # crossover_num = 25
        assert k in self.keep_top_k
        print('crossover ......')
        res = []
        max_iters = 10 * crossover_num

        def random_func():

            p1 = choice(self.keep_top_k[k])
            p2 = choice(self.keep_top_k[k])
            return tuple(choice([i, j]) for i, j in zip(p1, p2))
        cand_iter = self.stack_random_cand(random_func)
        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            # print('crossover {}/{}'.format(len(res), crossover_num))

        print('crossover_num = {}'.format(len(res)))
        return res

    def search(self):
        print('population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
            self.population_num, self.select_num, self.mutation_num, self.crossover_num, self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))


        print(f"loss limit:{self.loss_limit}")

        self.get_random(self.population_num)


        while self.epoch < self.max_epochs:
            print('epoch = {}'.format(self.epoch))
            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)


            self.update_top_k(self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['cost'])
            self.update_top_k(self.candidates, k=50, key=lambda x: self.vis_dict[x]['cost'])


            print('epoch = {} : top {} result'.format(self.epoch, len(self.keep_top_k[50])))
            for i, cand in enumerate(self.keep_top_k[50]):
                print('No.{} {} cost = {}, loss={}'.format(i + 1, cand, self.vis_dict[cand]['cost'], self.vis_dict[cand]['loss']))
                if i >2:
                    break

            mutation = self.get_mutation(self.select_num, self.mutation_num, self.m_prob)
            crossover = self.get_crossover(self.select_num, self.crossover_num)


            self.candidates = mutation + crossover
            self.get_random(self.population_num)
            self.epoch += 1



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=str, default='log')
    parser.add_argument('--max-epochs', type=int, default=30)
    parser.add_argument('--select-num', type=int, default=20)
    parser.add_argument('--population-num', type=int, default=50)
    parser.add_argument('--m_prob', type=float, default=0.1)
    parser.add_argument('--crossover-num', type=int, default=25)
    parser.add_argument('--mutation-num', type=int, default=25)
    parser.add_argument('--max-train-iters', type=int, default=200)
    parser.add_argument('--max-test-iters', type=int, default=40)
    parser.add_argument('--train-batch-size', type=int, default=128)
    parser.add_argument('--test-batch-size', type=int, default=200)
    parser.add_argument('--search_file', type=str, default='exp_dis_nas/logs/cifar_dis/search.log')
    parser.add_argument('--nr_layer', type=int, default=44)
    parser.add_argument('--nr_state', type=int, default=3)


    args = parser.parse_args()
    t = time.time()
    searcher = EvolutionSearcher(args)
    searcher.search()
    print('total searching time = {:.2f} hours'.format((time.time() - t) / 3600))

if __name__ == '__main__':
    try:
        main()
        os._exit(0)
    except:

        import traceback
        traceback.print_exc()
        time.sleep(1)
        os._exit(1)