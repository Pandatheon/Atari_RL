"""
    Deep Q Learning based on paper:

"""
import numpy as np
import random
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import v2
import gym

from utilis import initialize_weights
import Config


class CustomCropTransform:
    def __call__(self, img:torch.tensor):
        # roughly captures the playing area 84x84
        return v2.functional.crop(img, top=img.shape[1]-84, left=0, height=84, width=84)

class replay_memory:
    def __init__(self,Capacity):
        self.preprocessor = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float),
            v2.Grayscale(),
            v2.Resize(size=(110,84)),
            CustomCropTransform()
        ])
        self.pool = []
        self.capacity = Capacity
        self.temp = []

    def add(self, state, action, reward, next_state, done):
        state = self.preprocessor(state)
        next_state = self.preprocessor(next_state)
        self.temp.append((state, action, reward, next_state, done))
        # preprocessing last 4 frames and stack them into pool
        if len(self.temp) == 4:
            state, action, reward, next_state, done = zip(*self.temp)
            state = torch.stack(state,dim=1)
            next_state = torch.stack(next_state,dim=1)
            reward = reward[3]
            done = done[3]
            action = action[3]

            self.pool.append((state, action, reward, next_state, done))
            self.temp.clear()

        if len(self.pool) > self.capacity:
            self.pool.pop(0)

    def sample(self, batch_size):
        replays = random.sample(self.pool, batch_size)
        return replays

    def __len__(self):
        return len(self.pool)

class Q_net(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.conv1 = nn.Conv2d(self.in_channel, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.linear1 = nn.Linear(3136, 512)
        self.linear2 = nn.Linear(512, self.out_channel)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.linear1(x.flatten(1,3)))
        x = F.relu(self.linear2(x))
        return x

class DQNAgent:
    def __init__(self,
                 env: gym.Env,
                 discount_factor: float,
                 epsilon: list,
                 learning_rate: float,
                 packed : int,
                 target_update : int,
                 ):

        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon_list = epsilon

        self.net = Q_net(packed, env.action_space.n).apply(initialize_weights).to(Config.device)
        self.target_net = Q_net(packed, env.action_space.n).apply(initialize_weights).to(Config.device)
        self.Optimizer = torch.optim.RMSprop(self.net.parameters(), self.lr)
        self.updates = target_update
        self.episode = 0

    def take_action(self, state):
        # epsilon-greedy algorithm as Policy
        # annealed linearly from 1 to 0.1
        epsilon = (1-self.episode*10/Config.episodes)*\
                  (self.epsilon_list[0]-self.epsilon_list[1])+self.epsilon_list[1]
        if np.random.random() < epsilon:
            action = np.random.randint(self.env.action_space.n)
        else:
            state = state.to(Config.device)
            action = self.net(state).argmax().item()
        return action

    def update(self, replays):
        state, action, reward, next_state, done = zip(*replays)
        state = torch.concat(state,dim=0).to(Config.device)
        next_state = torch.concat(next_state,dim=0).to(Config.device)

        # a separate network in the Q-learning update.
        Q_values = self.net(state)[torch.arange(self.net(state).size(0)), action]
        Q_next_values = self.target_net(next_state).max(dim=1).values
        Q_target = torch.tensor(reward,dtype=torch.float).to(Config.device) +\
                   self.gamma * Q_next_values*(1 - torch.tensor(done,dtype=torch.float).to(Config.device))
        DQN_loss_sqrt = Q_target-Q_values

        # error clipping further improved the stability of the algorithm
        DQN_loss_sqrt = torch.clamp(DQN_loss_sqrt, -1, 1)
        DQN_loss = torch.square(DQN_loss_sqrt).mean()

        self.Optimizer.zero_grad()
        DQN_loss.backward()
        self.Optimizer.step()

        if self.episode % self.updates == 0:
            self.target_net.load_state_dict(self.net.state_dict())

