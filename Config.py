import torch

# episodes
seed = 0
frames = int(1e+7)
threshold = int(5e+4)
exploration_stop = int(1e+5)
discount_factor = 0.99
epsilon = [1, 0.1]
buffer_size = int(5e+4)
interval = 50000

# network training params
input_channel = 4  # 4 frames as 1 input in memory pool
lr = 0.00025
batch_size = 32
updates = int(1e+4)
save_path = r"experiment"

# hardware
device = 'cuda' if torch.cuda.is_available() else 'cpu'