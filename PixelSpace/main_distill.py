from main import parse_args_and_config
import argparse
import traceback    #主要用于在程序发生异常时，格式化和提取堆栈信息。这在调试和记录错误日志时非常有用。
import shutil  #用于实现文件的复杂操作，如复制、删除、移动
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb   #用于记录和可视化训练过程的指标（如损失、精度）
from runners.diffusion import Diffusion

#sci_mode=False 禁止科学记数法显示张量的值，确保以普通浮点格式打印
torch.set_printoptions(sci_mode=False)  #科学计数法：scientific notation


def main():
    args, config = parse_args_and_config()   #调用函数 parse_args_and_config()，解析命令行参数和配置文件
    #记录日志信息，进程的唯一标识符，实验的注释
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))

    runner = Diffusion(args, config)
    #架构搜索阶段
    if args.nas_search:
        runner.search_best()
    #采样阶段
    elif args.sample:
        runner.sample(stand_alone_sample=args.stand_alone_sample)
    #初始超网训练阶段
    elif args.super_train:
        runner.train_supernet()
    #子网重训练阶段
    elif args.stand_alone_train:
        runner.train_subnet()
    return 0

if __name__ == "__main__":
    sys.exit(main())     #main()的返回结果是0表示程序正常结束，然后程序结束运行
