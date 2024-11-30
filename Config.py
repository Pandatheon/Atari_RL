import torch

# episodes and net
seed = 42
pretrained = None
episodes = 500
threshold = 50
discount_factor = 0.99
epsilon = [1, 0.3]
pool_cap = 1000
packed = 4  # 4 frames as 1 input in memory pool

# network training params
epochs = 500
lr = 0.008
batch_size = 32
updates = 100

# hardware
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cudnn_benchmark = True