import os

import data
import torch
import logging
import itertools
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import defaultdict
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from datasets import get_dataset, data_transform, inverse_data_transform
from models.diffusion_teacher import Teacher

class ArchitectureIndividual:
    def __init__(self, encoding):
        self.encoding = encoding
        self.objectives = {}
        self.rank = 0
        self.distance = 0

    def __lt__(self, other):
        dominates = True
        for key in ['loss', 'params', 'latency']:
            if self.objectives[key] > other.objectives[key]:
                dominates = False
                break
        return dominates

class MONASProblem(Problem):
    def __init__(self, config, teacher, dataset, device):
        super().__init__(
            n_var=10,
            n_obj=3,
            n_constr=0,
            xl=0,
            xu=1
        )
        self.config = config
        self.teacher = teacher
        self.dataset = dataset
        self.device = device

        self.supernet = self.build_supernet()

        self.cache_teacher_guidance()

    def cache_teacher_guidance(self):
        self.test_x, self.test_guides, self.test_guide_hs, self.ts = [], [], [], []
        train_loader = data.DataLoader(self.dataset, batch_size=self.config.nas_search.batch_size, shuffle=True)

        with torch.no_grad():
            for i, (x, _) in enumerate(train_loader):
                if i >= self.config.nas_search.test_num - 1:
                    break

    def _evaluate(self, X, out, *args, **kwargs):
        obj_list = []
        for x in X:
            encoding = self.vector_to_architecture(x)

            subnet = self.supernet.sample_subnet(encoding)
            objectives = self.evaluate_subnet(subnet)

            obj_list.append([objectives['loss'],
                             -objectives['params'],
                             -objectives['latency']])

        out["F"] = np.array(obj_list)

    def evaluate_subnet(self, subnet):
        total_loss = 0
        for i in range(len(self.test_x)):
            x = self.test_x[i].to(self.device)
            t = self.ts[i].to(self.device)
            guides = self.test_guides[i].to(self.device)
            guide_hs = self.test_guide_hs[i].to(self.device)

            with torch.no_grad():
                pred = subnet(x, t, start_stage=0, forward_op=None)
                mse_loss = torch.mean((pred - guides) ** 2)

            total_loss += mse_loss.item()

        params = sum(p.numel() for p in subnet.parameters())
        latency = self.measure_latency(subnet)

        return {
            'loss': total_loss / len(self.test_x),
            'params': params,
            'latency': latency
        }

    def measure_latency(self, model):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        dummy_input = torch.randn(1, 3, 32, 32).to(self.device)

        starter.record()
        _ = model(dummy_input)
        ender.record()
        torch.cuda.synchronize()
        return starter.elapsed_time(ender)

def moea_search(config, args, teacher, dataset, device):
    problem = MONASProblem(config, teacher, dataset, device)

    algorithm = NSGA2(
        pop_size=50,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15, vtype=int),
        mutation=PM(prob=0.1, eta=20),
        eliminate_duplicates=True
    )

    result = minimize(
        problem,
        algorithm,
        ('n_gen', 100),
        seed=42,
        verbose=True
    )

    pareto_front = result.F
    pareto_archs = [problem.vector_to_architecture(x) for x in result.X]

    visualize_pareto_front(pareto_front)
    save_pareto_solutions(pareto_archs, config)

    return pareto_archs

def visualize_pareto_front(F):
    plt.figure(figsize=(10, 6))
    ax = plt.axes(projection='3d')
    ax.scatter3D(-F[:, 0], -F[:, 1], -F[:, 2], c='r')
    ax.set_xlabel('Loss')
    ax.set_ylabel('Params')
    ax.set_zlabel('Latency')
    plt.show()

def save_pareto_solutions(pareto_archs, config):
    save_dir = os.path.join(config.args.log_path, "pareto_solutions")
    os.makedirs(save_dir, exist_ok=True)

    csv_path = os.path.join(save_dir, "pareto_front.csv")
    with open(csv_path, 'w') as f:
        f.write("arch_id,encoding,loss,params,latency\n")
        for idx, arch in enumerate(pareto_archs):
            encoding_str = '-'.join(map(str, arch.encoding))
            line = f"{idx},{encoding_str},{arch.objectives['loss']:.4f},{arch.objectives['params']},{arch.objectives['latency']:.2f}\n"
            f.write(line)

    json_path = os.path.join(save_dir, "details.json")
    details = {
        "config": vars(config),
        "solutions": [
            {
                "id": idx,
                "encoding": arch.encoding.tolist() if isinstance(arch.encoding, np.ndarray) else arch.encoding,
                "loss": arch.objectives['loss'],
                "params": arch.objectives['params'],
                "latency": arch.objectives['latency']
            }
            for idx, arch in enumerate(pareto_archs)
        ]
    }
    import json
    with open(json_path, 'w') as f:
        json.dump(details, f, indent=2)

    logging.info(f"已保存Pareto前沿方案至: {save_dir}")

class MONASProblem(Problem):
    def vector_to_architecture(self, x):
        int_encoding = np.round(x).astype(int)
        return ArchitectureIndividual(encoding=int_encoding)

class NAS_Searcher:
    def __init__(self, config, args, device):
        self.config = config
        self.args = args
        self.device = device
        self.betas = torch.linspace(config.model.beta_start, config.model.beta_end, config.model.num_timesteps).to(
            device)

    def search_best(self):
        teacher = self.load_teacher()

        dataset, _ = get_dataset(self.args, self.config)

        pareto_archs = moea_search(self.config, self.args, teacher, dataset, self.device)

        best_arch = min(pareto_archs, key=lambda x: x.objectives['latency'])
        return best_arch.encoding

    def load_teacher(self):
        teacher = Teacher(self.config)
        return teacher

class SuperStudent(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.build_search_space()

    def build_search_space(self):
        self.layers = nn.ModuleList()
        for _ in range(self.config.nas_search.max_layers):
            layer = nn.ModuleDict({
                'conv3x3': nn.Conv2d(64, 64, 3, padding=1),
                'conv5x5': nn.Conv2d(64, 64, 5, padding=2),
                'skip': nn.Identity()
            })
            self.layers.append(layer)

    def sample_subnet(self, encoding):
        class SubNet(nn.Module):
            def __init__(self, parent, encoding):
                super().__init__()
                self.layers = []
                for i, op_idx in enumerate(encoding):
                    layer = parent.layers[i]
                    if op_idx == 0:
                        self.layers.append(layer['conv3x3'])
                    elif op_idx == 1:
                        self.layers.append(layer['conv5x5'])
                    else:
                        self.layers.append(layer['skip'])
                self.layers = nn.Sequential(*self.layers)

            def forward(self, x, t, **kwargs):
                return self.layers(x)

        return SubNet(self, encoding)
