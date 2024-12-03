from torch import nn
import torch
import numpy as np
import os
import logging
import yaml
import datetime
import random



def ensure_repo(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def save_params(args):
    args.exp_name = args.save_dir + "_" + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')

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

    torch.save(agent.Optimizer.state_dict(),
               os.path.join("exp", args.exp_name, 'E{}_optimizer.pt'.format(agent.frame_count)))
    torch.save(agent.net.state_dict(),
               os.path.join("exp", args.exp_name, 'E{}_net.pt'.format(agent.frame_count)))
    torch.save(agent.target_net.state_dict(),
               os.path.join("exp", args.exp_name, '{}_target_net.pt'.format(agent.frame_count)))
    state, action, reward, next_state, done = zip(*memory_pool)
    state = torch.concat(state, dim=0)
    next_state = torch.concat(next_state, dim=0)
    action = torch.tensor(reward).view(-1, 1)
    reward = torch.tensor(reward).view(-1, 1)
    done = torch.tensor(done).view(-1, 1)
    torch_dict={"state": state,
                "next_state": next_state,
                "action": action,
                "reward": reward,
                "done": done}
    torch.save(torch_dict, os.path.join("exp", args.exp_name, '{}_buffer.pt'.format(agent.frame_count)))

def continue_checkpoint(args, agent, memory_pool, frame_count):
    agent.net.load_state_dict(os.path.join("exp", args.exp_name, 'net.pt'))
    agent.target_net.load_state_dict(os.path.join("exp", args.exp_name, 'target_net.pt'))
    agent.Optimizer.load_state_dict(os.path.join("exp", args.exp_name, 'optimizer.pt'))
    agent.frame_count = frame_count

