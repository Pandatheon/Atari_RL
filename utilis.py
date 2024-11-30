from torch import nn
import torch
import numpy as np
import os
import logging
import yaml
import datetime
import sys

def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def ensure_repo(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def save_params(args):
    args.exp_name = args.save_dir + "_" + datetime.datetime.now().strftime("%mM_%dD_%HH") + "_" + \
                    "{:04d}".format(np.random.randint(0, 1000))

    # make file
    if not os.path.exists(os.path.join("exp", args.exp_name)):
        os.makedirs(os.path.join("exp", args.exp_name))

    # yaml file
    with open(os.path.join("exp", args.exp_name, "config.yml"), "w") as f:
        yaml.dump(args, f)
    f.close()

def create_logger(logger_file_path):
    if not os.path.exists(logger_file_path):
        os.makedirs(logger_file_path)
    log_name = '{}.log'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))
    final_log_file = os.path.join(logger_file_path, log_name)

    logger = logging.getLogger()  # 设定日志对象
    logger.setLevel(logging.INFO)  # 设定日志等级

    file_handler = logging.FileHandler(final_log_file)  # 文件输出
    console_handler = logging.StreamHandler()  # 控制台输出

    # 输出格式
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s "
    )

    file_handler.setFormatter(formatter)  # 设置文件输出格式
    console_handler.setFormatter(formatter)  # 设施控制台输出格式
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def save_checkpoint(args, agent, memory_pool):
    with open(os.path.join("exp", args.exp_name, "pool.txt"), "w") as file:
        for tpl in memory_pool.pool:
            file.write(",".join(map(str, tpl)) + "\n")
    torch.save(agent.net.state_dict(), os.path.join("exp", args.exp_name, 'E{}_net.pt'.format(agent.episode)))
    torch.save(agent.target_net.state_dict(), os.path.join("exp", args.exp_name, '{}_target_net.pt'.format(agent.episode)))

def continue_checkpoint(args,agent,memory_pool):
    agent.net.load_state_dict(os.path.join("exp", args.exp_name, 'net.pt'))
    agent.target_net.load_state_dict(os.path.join("exp", args.exp_name, 'target_net.pt'))
    # agent.episode =
    with open(os.path.join("exp", args.exp_name, "pool.txt"), "w") as file:
        for line in file:
            memory_pool.pool.append(tuple(map(int, line.strip().split(","))))
    file.close()
