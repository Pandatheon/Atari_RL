import argparse
import torch

parser = argparse.ArgumentParser("Deep_Q_learning")
parser.add_argument('--frames', type=int,
                    default=int(1e+7), help='Numbers of frames')
parser.add_argument('--discount', type=float,
                    default=0.99, help='Gamma in formula')
parser.add_argument('--epsilon_init', type=float,
                    default=1, help='initial epsilon for policy')
parser.add_argument('--epsilon_last', type=float,
                    default=0.1, help='epsilon')
parser.add_argument('--target_update', type=int,
                    default=int(1e+4), help='Gap for target network')
parser.add_argument('--online_update', type=int,
                    default=4, help='Gap for online network')
parser.add_argument('--observance_stop', type=int,
                    default=int(5e+4), help='frame of no observation')
parser.add_argument('--exploration_stop', type=int,
                    default=int(1e+6), help='frame of no exploration')

parser.add_argument('--buffer_size', type=int,
                    default=int(1e+5), help='Capacity of memory pool')
parser.add_argument('--input_channel', type=float,
                    default=4, help='Number of channels of input')
parser.add_argument('--batch_size', type=int,
                    default=32, help='batch size')
parser.add_argument('--learning_rate', type=float,
                    default=0.00025, help='init learning rate')

parser.add_argument('--interval', type=int,
                    default=int(2e+6), help='interval for check point')
parser.add_argument('--seed', type=int,
                    default=0, help='seed')
parser.add_argument('--device', type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu', help="CUDA oder CPU")
parser.add_argument('--save_dir', type=str,
                    default=r"experiment", help="save exp folder name")
args = parser.parse_args()