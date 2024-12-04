from torch import nn
import torch
import numpy as np
import os
import logging
import yaml
import datetime
import random
import glob
import shutil

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

def create_exp_dir(path, scripts_to_save=None):
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'Codes')):
            os.mkdir(os.path.join(path, 'Codes'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'Codes', os.path.basename(script))
            shutil.copyfile(script, dst_file)

    with open(os.path.join(path,"_description.txt"),'a') as f:
        f.write("This experiment was for: \n")
        f.write("\nThis experiment ended with: \n")
    f.close()

def create_logger(logger_file_path):
    if not os.path.exists(logger_file_path):
        os.makedirs(logger_file_path)
    log_name = '{}.log'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))
    final_log_file = os.path.join(logger_file_path, log_name)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(final_log_file)
    console_handler = logging.StreamHandler()

    # 输出格式
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s "
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def save_params(args):
    args.exp_name = args.save_dir + "_" + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')

    # make file
    if not os.path.exists(os.path.join("exp", args.exp_name)):
        os.makedirs(os.path.join("exp", args.exp_name))

    # yaml file
    with open(os.path.join("exp", args.exp_name, "config.yml"), "w") as f:
        yaml.dump(args, f)
    f.close()

    create_exp_dir(os.path.join("exp", args.exp_name),
               scripts_to_save=glob.glob('*.py'))


def save_checkpoint(args, agent):
    torch.save(agent.net.state_dict(),
               os.path.join("exp", args.exp_name, 'E{}_net.pt'.format(agent.frame_count)))
    torch.save(agent.target_net.state_dict(),
               os.path.join("exp", args.exp_name, '{}_target_net.pt'.format(agent.frame_count)))
