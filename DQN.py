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
from collections import deque
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
        self.pool = deque(maxlen=Capacity)
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

        self.conv1 = nn.Conv2d(self.in_channel, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.linear1 = nn.Linear(3136, 512)
        self.linear2 = nn.Linear(512, self.out_channel)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.linear1(x.flatten(1,3)))
        x = self.linear2(x)
        return x

class DQNAgent:
    def __init__(self,
                 env: gym.Env,
                 discount_factor: float,
                 epsilon: list,
                 exploration_stop: int,
                 input_channel: int,
                 learning_rate: float,
                 target_update: int,
                 device: str
                 ):

        self.env = env
        self.temp_frames = []
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon_list = epsilon
        self.input_channel = input_channel
        self.exploration_stop = exploration_stop

        self.device = device
        self.net = Q_net(input_channel, env.action_space.n).apply(initialize_weights).to(self.device)
        self.target_net = Q_net(input_channel, env.action_space.n).apply(initialize_weights).to(self.device)
        self.Optimizer = torch.optim.RMSprop(self.net.parameters(), self.lr)
        self.updates = target_update


        self.frame_count = 0
        self.update_count = 0

    def take_action(self, state):
        # epsilon-greedy algorithm as Policy
        # annealed linearly from 1 to 0.1
        decay = 1-(self.frame_count/self.exploration_stop) if 1-(self.frame_count/self.exploration_stop)>0 else 0
        epsilon = decay*(self.epsilon_list[0]-self.epsilon_list[1])+self.epsilon_list[1]

        # stack four into one
        self.temp_frames.append(state)
        if len(self.temp_frames) < self.input_channel:
            stacked_state = torch.stack([state] * self.input_channel, dim=1)  # compromise
        else:
            stacked_state = torch.stack(self.temp_frames, dim=1)  # stack recent 4
            self.temp_frames.pop(0)

        if np.random.random() < epsilon:
            action = np.random.randint(self.env.action_space.n)
        else:
            stacked_state = stacked_state.to(self.device)
            action = self.net(stacked_state).argmax().item()
        return action

    def update(self, replays):
        state, action, reward, next_state, done = zip(*replays)
        state = torch.concat(state, dim=0).to(self.device)
        next_state = torch.concat(next_state, dim=0).to(self.device)
        action = torch.tensor(reward,dtype=torch.int64).view(-1,1).to(self.device)
        reward = torch.tensor(reward,dtype=torch.float).view(-1,1).to(self.device)
        done = torch.tensor(done, dtype=torch.float).view(-1,1).to(self.device)

        # a separate network in the Q-learning update.
        Q_values = self.net(state).gather(1, action)
        Q_next_values = self.target_net(next_state).max(dim=1).values.view(-1,1)
        Q_target = reward + self.gamma * Q_next_values * (1 - done)
        DQN_loss = F.huber_loss(Q_values, Q_target)

        self.Optimizer.zero_grad()
        DQN_loss.backward()
        self.Optimizer.step()

        if self.update_count % self.updates == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        self.update_count += 1
