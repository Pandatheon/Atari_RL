import torch

# episodes and net
seed = 42
episodes = 50000
threshold = 250
discount_factor = 0.99
epsilon = [1, 0.1]
pool_cap = 5000
packed = 4  # 4 frames as 1 input in memory pool

# network training params
lr = 0.0005
batch_size = 32
updates = 50
save_path = r".\experiment"

# hardware
device = 'cuda' if torch.cuda.is_available() else 'cpu'