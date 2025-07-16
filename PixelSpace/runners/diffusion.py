import os
import logging
import time
import math
import glob #导入glob模块，支持文件路径模式匹配，用于查找符合特定规则的文件路径。
import numpy as np
import tqdm
import torch
import torch.utils.data as data
from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import *
from datasets import get_dataset, data_transform, inverse_data_transform
#从functions.ckpt_util模块导入get_ckpt_path函数，用于获取或管理模型的检查点路径
from functions.ckpt_util import get_ckpt_path
from functions.denoising import generalized_steps
from functions.denoising import ddpm_steps, dpm_solver_steps, pndm_steps
#导入torchvision.utils并为其指定别名tvu，这是一个常用于图像处理的库，提供如保存图像、创建网格图像等功能。
import torchvision.utils as tvu
from models.diffusion_teacher import Teacher
from models.diffusion_student import Student
from models.student_supernet import SuperStudent, StandAloneNet
#导入itertools库，用于处理迭代器（如生成排列组合等）
import  itertools
from models.student_supernet import KERNEL_SIZE, CHANNEL_NUM
import pickle
from search_cons import search_best_local
#用于在多GPU训练时平衡数据的并行处理。
from functions.data_balance_parallel import BalancedDataParallel
from .model_id import id2ops


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):

    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )

    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )

    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)

    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        #，np.linspace(start, stop, num, dtype) 用于在 [start, stop] 范围内生成 num 个等间隔的数值。
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)

    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def search_best(file):
    pass


class Diffusion(object):

    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        #设置设备
        if device is None:
            device = (torch.device("cuda")if torch.cuda.is_available()else torch.device("cpu"))
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod =self.alphas_cumprod= alphas.cumprod(dim=0)

        alphas_cumprod_prev = torch.cat([torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0)
        posterior_variance = (betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":

            self.logvar = posterior_variance.clamp(min=1e-20).log()

    # train the base or searched model without distillation
    def train(self, use_nas_result=False):

        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.retraining.batch_size//config.retraining.grad_acc_steps,
            shuffle=True,
            num_workers=config.data.num_workers,

            drop_last= True
        )

        # teacher and  model
        if not use_nas_result:
            model = Model(config)
        else:
            forward_op = search_best(file='/data/JiaoChunXiao/LDMOES/PixelSpace/exp_dis_nas/logs/cifar_dis/search_8.log')
            model = StandAloneNet(config, forward_op)

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        optimizer = get_optimizer(self.config,  model.parameters() )
       
        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:

            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))

            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        start_step = step

        acc_loss = 0
        for epoch in range(start_epoch, self.config.retraining.n_epochs):
            data_start = time.time()
            data_time = 0

            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1
                real_step = (step-1-start_step)//(config.retraining.grad_acc_steps) + 1

                x = x.to(self.device)
                x = data_transform(self.config, x)

                e = torch.randn_like(x)
                b = self.betas

                t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

                loss = loss_registry[config.model.type](model, x, t, e, b)

                loss = loss/config.retraining.grad_acc_steps
                acc_loss += loss.item()
                loss.backward()

                if step % config.retraining.grad_acc_steps == 0:
                    try:
                        torch.nn.utils.clip_grad_norm_( model.parameters(), config.optim.grad_clip ) #梯度剪裁操作，防止梯度爆炸现象
                    except Exception:
                        pass
                    optimizer.step()
                    logging.info(f"step: {step}, loss: {acc_loss}" )
                    acc_loss = 0
                    if self.config.model.ema:
                        ema_helper.update(model)
                    optimizer.zero_grad()
                if real_step % self.config.retraining.snapshot_freq == 0:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())
                    torch.save(states,os.path.join(self.args.log_path, "ckpt_{}.pth".format(real_step)),)

                data_start = time.time()

    #采样过程
    def sample(self, stand_alone_sample=False):
 

        if stand_alone_sample:
            # forward_op = search_best(file='exp_dis_nas/logs/cifar_dis/search.log')
            # forward_op=[1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1]

            # forward_op = [1, 0, 1, 0, 1, 0, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 0, 1, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  
            # id 7
            # forward_op = [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 2, 0, 1, 1, 1, 1, 2, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1]
            # id 8
            # forward_op =  [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 2, 0, 1, 1, 1, 1, 2, 1, 1, 0, 2, 0, 1, 0, 0, 0, 0, 0, 1]

            # id 3
            # forward_op = [1, 0, 1, 0, 1, 0, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 0, 1, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
            # id 6
            # forward_op = [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 2, 0, 0, 1, 1, 1, 2, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1]
            # id 9
            # forward_op = [0, 1, 2, 0, 1, 0, 1, 0, 1, 1, 0, 2, 0, 0, 1, 1, 0, 0, 1, 0, 2, 2, 1, 2, 2, 0, 0, 1, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
            # cele 1
            # forward_op = [1, 0, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0]
            # forward_op = id2ops["church_1"]
            forward_op = id2ops["cifar10_searched"]
            model = StandAloneNet(self.config, forward_op)
        else:
            model =  Model(self.config)


        if not self.args.use_pretrained:

            if getattr(self.config.sampling, "ckpt_id", None) is None:
                states = torch.load(
                    os.path.join(self.args.log_path, "ckpt.pth"),
                    map_location=self.config.device,
                )
            else:

                ckpt_path = os.path.join(self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth")
                states = torch.load(ckpt_path, map_location=self.config.device)
                logging.warn(f"load checkpoint from {ckpt_path}")
                
            model = model.to(self.device)
            model = torch.nn.DataParallel(model)

            model.load_state_dict(states[0], strict=True)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None
        else:

                if self.config.data.dataset == "CIFAR10":
                    name = "cifar10"
                elif self.config.data.dataset == "LSUN":
                    name = f"lsun_{self.config.data.category}"
                else:
                    raise ValueError
                ckpt = get_ckpt_path(f"ema_{name}")

            print("Loading checkpoint {}".format(ckpt))
                model.load_state_dict(torch.load(ckpt, map_location=self.device))
                model.to(self.device)
                model = torch.nn.DataParallel(model)

        model.eval()
        

        if self.args.fid:
            self.sample_fid(model)
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence:
            self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")


    def sample_fid(self, model):
        config = self.config

        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"starting from image {img_id}")
        # total_n_samples = 50000
        total_n_samples = self.config.sampling.total_num

        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size

        with torch.no_grad():
            for _ in tqdm.tqdm(range(n_rounds), desc="Generating image samples for FID evaluation."):

                n = config.sampling.batch_size

                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                x = self.sample_image(x, model)
                x = inverse_data_transform(config, x)

                for i in range(n):
                    tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{img_id}.png"))
                    img_id += 1

    def sample_sequence(self, model):
        config = self.config
        x = torch.randn(
            8,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )


        with torch.no_grad():
            _, x = self.sample_image(x, model, last=False)

        x = [inverse_data_transform(config, y) for y in x]

        for i in range(len(x)):
            for j in range(x[i].size(0)):
                tvu.save_image(
                    x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
                )

    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i : i + 8], model))
        x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))


    def sample_image(self, x, model, last=True):

        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == 'dpm':
            x = dpm_solver_steps(model,x, self.betas, self.args.timesteps, self.args.skip_type)

        elif self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (np.linspace(0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps) ** 2 )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta,)
            x = xs

        elif self.args.sample_type == "pndm":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (np.linspace(0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps)** 2)
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            x = pndm_steps(model, x, seq, self.alphas_cumprod)
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (np.linspace(0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps) ** 2)
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError

        if last:
            x = x[0][-1]
        return x

    def test(self):
        pass


    def train_subnet(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.retraining.batch_size//config.retraining.grad_acc_steps,
            shuffle=True,
            num_workers=config.data.num_workers,
        )

        # forward_op = search_best(file='exp_dis_nas/logs/cifar_dis/search.log')
        # id3 EA
        # forward_op = [1, 0, 1, 0, 1, 0, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 0, 1, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  
        # forward_op = [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1]
        # id 7
        # forward_op = [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 2, 0, 1, 1, 1, 1, 2, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1]
        # id8
        # forward_op =  [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 2, 0, 1, 1, 1, 1, 2, 1, 1, 0, 2, 0, 1, 0, 0, 0, 0, 0, 1]
        # id 9
        # forward_op = [0, 1, 2, 0, 1, 0, 1, 0, 1, 1, 0, 2, 0, 0, 1, 1, 0, 0, 1, 0, 2, 2, 1, 2, 2, 0, 0, 1, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
        #  cele id1
        # forward_op = [1, 0, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0]
        # cele test
        # forward_op = [0, 0, 2, 0, 1, 0, 0, 1, 0, 0, 0, 0, 2, 2, 2, 0, 0, 1, 1, 1, 0, 1, 1, 2, 1, 2, 2, 2, 2, 0, 2, 1, 1, 2, 0, 2, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 2, 1, 0, 1, 1, 1, 0, 0]
        # teacher and student model
        # forward_op = id2ops["church_1"]
        forward_op = id2ops["cifar10_first"]
        model = StandAloneNet(config, forward_op)
        print("create teacher")

        teacher = Teacher(config)
        # exit(-1)

        if self.config.data.dataset == "CELEBA" or self.config.data.own:
            if self.config.data.dataset == "CELEBA":
                states = torch.load('celeba.pth')
            else:
                states = torch.load('exp_church_base_128/logs/church/ckpt_450000.pth')
            teacher =  teacher.to(self.device)
            teacher = torch.nn.DataParallel(teacher)
            teacher.load_state_dict(states[0])
            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(teacher)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(teacher)
            else:
                ema_helper = None
            del states
        else:
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else: 
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            teacher.load_state_dict(torch.load(ckpt, map_location=self.device))
            teacher= teacher.to(self.device)
            teacher= torch.nn.DataParallel(teacher)

        model= model.to(self.device)
        model =  torch.nn.DataParallel(model)
        teacher.eval()
        model.train()

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        real_step = 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])
            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            real_step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        acc_loss = 0
        step = 0
        for epoch in range(config.retraining.n_epochs):
            logging.info(f"---------------------epoch:{epoch}---------------------")
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                step += 1

                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

                forward_op = None
                stage = 0

                #alpha = min(1, real_step *(1/config.retraining.alpha_step) )

                #alpha = 1 if real_step >= config.retraining.alpha_step else 0

                alpha = 0.5 * (1 + math.cos(math.pi * real_step / config.retraining.alpha_step)) if real_step < config.retraining.alpha_step else 1

                scale = self.config.retraining.alpha_scale

                loss = distill_loss(teacher, model, x, t, e, b, start=stage, train_sub=True, alpha=alpha, alpha_scale = scale)
                tb_logger.add_scalar("loss", loss, global_step=step)

                # optimizer.zero_grad()
                loss = loss/config.retraining.grad_acc_steps
                acc_loss += loss.item()
                loss.backward()

                if step % config.retraining.grad_acc_steps == 0:
                    real_step +=1
                    try:
                        torch.nn.utils.clip_grad_norm_( model.parameters(), config.optim.grad_clip )
                    except Exception:
                        pass
                    optimizer.step()
                    optimizer.zero_grad()
                    logging.info(f"step: {real_step}, loss: {acc_loss}" )
                    acc_loss = 0
                    if self.config.model.ema:
                        ema_helper.update(model)
                
                if real_step % self.config.retraining.snapshot_freq == 0 or  real_step == self.config.retraining.alpha_step:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        real_step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())
                    torch.save(states,os.path.join(self.args.log_path, "ckpt_{}.pth".format(real_step)),)

        # save the last one
        states = [
            model.state_dict(),
            optimizer.state_dict(),
            epoch,
            real_step,
        ]
        if self.config.model.ema:
            states.append(ema_helper.state_dict())

        torch.save( states, os.path.join(self.args.log_path, "ckpt_{}.pth".format(real_step)))


    def train_supernet(self):
        
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.nas_training.batch_size//config.nas_training.grad_acc_steps,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        # teacher and student model
        model = SuperStudent(config)
        teacher = Teacher(config)
        # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
        if self.config.data.dataset == "CELEBA" or self.config.data.own:
            if self.config.data.dataset == "CELEBA":
                states = torch.load('celeba.pth')
            else:
                states = torch.load('exp_church_base_128/logs/church/ckpt_450000.pth')
            teacher =  teacher.to(self.device)
            # teacher = torch.nn.DataParallel(teacher)
            teacher = BalancedDataParallel(1, teacher, dim=0).cuda()

            teacher.load_state_dict(states[0])
            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(teacher)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(teacher)
            else:
                ema_helper = None
        else:
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else: 
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            teacher.load_state_dict(torch.load(ckpt, map_location=self.device))
            teacher= teacher.to(self.device)
            teacher= torch.nn.DataParallel(teacher)


        model= model.to(self.device)
        # model = torch.nn.DataParallel(model)
        model = BalancedDataParallel(1, model, dim=0).cuda()


        teacher.eval()
        model.train()
        optimizer = get_optimizer(self.config, model.parameters())

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            del states


        stage_num =  2*len(config.model.ch_mult) +2

        num_res_blocks= self.config.model.num_res_blocks

        num_resolutions = len(self.config.model.ch_mult)

        layer_num = [num_res_blocks*2] * num_resolutions + [2*2] + [(num_res_blocks+1)*2] * num_resolutions + [0]
        for stage in range(0, stage_num-1):
            logging.info(f"---------------------stage:{stage}---------------------{stage_num}")
            step = 0
            optimizer.zero_grad()
            acc_loss = 0
            #

            # forward_op = np.random.randint(0,len(KERNEL_SIZE), layer_num[stage])
            kernel_index = list(np.random.randint(0,len(KERNEL_SIZE), layer_num[stage]))
            # ch_num_index = list(np.random.randint(0,len(CHANNEL_NUM), 1)) * layer_num[stage]

            ch_num_index = list(np.random.randint(0,len(CHANNEL_NUM), 1)) * (layer_num[stage]//2) + list(np.random.randint(0,len(CHANNEL_NUM), 1)) * (layer_num[stage]//2)

            forward_op = list(zip(kernel_index, ch_num_index))

            for epoch in range(config.nas_training.epoch):
                logging.info(f"---------------------epoch:{epoch}---------------------")
                for i, (x, y) in enumerate(train_loader):
                    n = x.size(0)
                    step += 1
                    x = x.to(self.device)
                    x = data_transform(self.config, x)
                    e = torch.randn_like(x)
                    b = self.betas
                    # antithetic sampling
                    t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                    t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                    # forward_op = np.random.randint(0,len(KERNEL_SIZE), layer_num[stage])
                    
                    loss = distill_loss(teacher, model, x, t, e, b, start=stage, forward_op=forward_op )
                    tb_logger.add_scalar("loss", loss, global_step=step)

                    # ops = " ".join('%s' %op for op in forward_op)
                    # logging.info(
                    #     f"step: {step}, loss: {loss.item()}, ops: {ops}  "
                    # )

                    # optimizer.zero_grad()
                    loss = loss/config.nas_training.grad_acc_steps
                    acc_loss += loss.item()
                    loss.backward()

                    # try:
                    #     torch.nn.utils.clip_grad_norm_(
                    #         model.parameters(), config.optim.grad_clip
                    #     )
                    # except Exception:
                    #     pass

                    if step % config.nas_training.grad_acc_steps == 0:
                        try:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_clip)
                        except Exception:
                            pass
                        optimizer.step()
                        optimizer.zero_grad()
                        # ops = " ".join('%s' %op for op in forward_op)
                        logging.info(f"step: {step}, loss: {acc_loss}, ops: {forward_op} ")
                        acc_loss = 0
                        forward_op = np.random.randint(0,len(KERNEL_SIZE), layer_num[stage])
                        kernel_index = list(np.random.randint(0,len(KERNEL_SIZE), layer_num[stage]))
                        ch_num_index = list(np.random.randint(0,len(CHANNEL_NUM), 1)) * (layer_num[stage]//2) + list(np.random.randint(0,len(CHANNEL_NUM), 1)) * (layer_num[stage]//2)
                        forward_op = list(zip(kernel_index, ch_num_index))

                    # optimizer.step()
                    if step > config.nas_training.steps:
                        break

          
            states = [
                model.state_dict(),
                optimizer.state_dict(),
                epoch,
                step,
            ]
            torch.save(states,os.path.join(self.args.log_path, "ckpt_stage_{}.pth".format(stage)),)
        # save the last one
        states = [
            model.state_dict(),
            optimizer.state_dict(),
            epoch,
            step,
        ]
        # torch.save( states, os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)))


    def search_best(self):
        stage_num =  2*len(self.config.model.ch_mult) +2
        student = SuperStudent(self.config)


        teacher = Teacher(self.config)
        # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
        if self.config.data.dataset == "CELEBA" or self.config.data.own:
            if self.config.data.dataset == "CELEBA":
                states = torch.load('celeba.pth')
            elif self.config.data.dataset=="LSUN" and self.config.data.category == "church_outdoor":
                states = torch.load('exp_church_base_128/logs/church/ckpt_450000.pth')
            teacher =  teacher.to(self.device)
            teacher = torch.nn.DataParallel(teacher)
            teacher.load_state_dict(states[0])
            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(teacher)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(teacher)
            else:
                ema_helper = None
        else:
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            teacher.load_state_dict(torch.load(ckpt, map_location=self.device))
            teacher= teacher.to(self.device)
            teacher= torch.nn.DataParallel(teacher)
        teacher.eval()

        # search on train dataset
        dataset, test_dataset = get_dataset(self.args, self.config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=self.config.nas_search.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
        )

        def tensors_to(tensors, device='cpu'):
            n_tensors = [tensor.to(device) for tensor in tensors]
            return n_tensors


        with torch.no_grad():
            # cifar 10
            num_res_blocks= self.config.model.num_res_blocks
            num_resolutions = len(self.config.model.ch_mult)
            layer_num = [num_res_blocks*2] * num_resolutions + [2*2] + [(num_res_blocks+1)*2] * num_resolutions + [0]
            best_ops = []
            test_num = self.config.nas_search.test_num
            test_x = []

            test_guides = []
            test_guide_hs = []
            ts = []

            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas #b=β
                # antithetic sampling
                t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
                x = x * a.sqrt() + e * (1.0 - a).sqrt()
                guides, guide_hs = teacher(x, t.float())

                test_x.append(x.cpu())
                test_guides.append(tensors_to(guides,'cpu'))
                test_guide_hs.append(tensors_to(guide_hs,'cpu'))
                ts.append(t.cpu())
                if i >= test_num-1:
                    break

            student = student.to(self.device)
            student = torch.nn.DataParallel(student)

            # for stage in range(5, stage_num-1):
            # for stage in range(0,5):
            # for stage in range(7,stage_num-1):
            #for stage in [7]:

            for stage in [8]:
                logging.info(f"---------------------stage:{stage}---------------------")
                ckpt_path = os.path.join(self.args.log_path, f"ckpt_stage_{stage}.pth")
                states = torch.load(ckpt_path, map_location=self.config.device)
                logging.warn(f"load checkpoint from {ckpt_path}")
                student.load_state_dict(states[0], strict=True)
                student.eval()
                # model_pool = itertools.product(range(len(KERNEL_SIZE)),repeat=layer_num[stage])
                model_pool_tmp = itertools.product(range(len(KERNEL_SIZE)), range(len(CHANNEL_NUM)))
                model_pool_tmp = itertools.product(model_pool_tmp,repeat=layer_num[stage])
                # remove replicated
                model_pool = []
                for ops in model_pool_tmp:
                    ops = list(ops)
                    mark = False
                    for k in range(len(ops)//2):

                        if ops[2*k][1] != ops[2*k+1][1]:
                            mark = True
                    if not mark:
                        model_pool.append(ops)
                print(len(model_pool))

                for forward_op in model_pool:
                    loss = 0
                    relative_loss = 0
                    # for i, (x, y) in enumerate(train_loader):
                    for i, x in enumerate(test_x):
                        x = x.to(self.device)
                        t = ts[i].to(self.device)
                        guide = tensors_to(test_guides[i], self.device)
                        guide_hs = tensors_to(test_guide_hs[i],self.device)
                        mse_loss, r_loss= search_loss(teacher, student, x , t, start=stage, forward_op=forward_op,
                                            guides=guide, guide_hs=guide_hs)
                        loss+=mse_loss
                        relative_loss+=r_loss

                    loss = loss/(i+1)
                    relative_loss = relative_loss/(i+1)
                    # ops = " ".join('%s' %op for op in forward_op)
                    logging.info(f"ops:{forward_op}, loss={loss}, r_loss={relative_loss}")
        
