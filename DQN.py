"""
    Deep Q Learning based on paper:

"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import v2
import gym
from collections import deque
from utilis import initialize_weights


class CustomCropTransform:
    def __call__(self, img:torch.tensor):
        # roughly captures the playing area 84x84
        return v2.functional.crop(img, top=img.shape[1]-84, left=0, height=84, width=84)

class replay_memory:
    def __init__(self,capacity, input_channel):
        self.preprocessor = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8),
            v2.Grayscale(),
            v2.Resize(size=(110,84)),
            CustomCropTransform()
        ])
        self.capacity = capacity
        self.state_pool = torch.zeros([capacity,input_channel,84,84],dtype=torch.uint8)
        self.next_state_pool = torch.zeros([capacity,input_channel,84,84],dtype=torch.uint8)
        self.act_reward_done = torch.zeros([capacity,3])
        self.temp = deque(maxlen=input_channel)
        self.index = 0

    def add(self, state, action, reward, next_state, done):
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

            temp_index = (self.index) % self.capacity
            self.state_pool[temp_index] = state
            self.next_state_pool[temp_index] = next_state
            self.act_reward_done[temp_index,:] = torch.tensor([action, reward, done])
            self.index += 1


    def sample(self, batch_size):
        replays_index = np.random.randint(0, min(self.index, self.capacity), size=batch_size)
        replays = (self.state_pool[replays_index],
                   self.act_reward_done[replays_index][:,0],
                   self.act_reward_done[replays_index][:,1],
                   self.next_state_pool[replays_index],
                   self.act_reward_done[replays_index][:,2])
        return replays

    def __len__(self):
        return min(self.index, self.capacity)

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
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        x = F.relu(self.linear1(x.flatten(1,3)), inplace=True)
        x = self.linear2(x)
        return x

class DQNAgent:
    def __init__(self,
                 env: gym.Env,
                 discount_factor: float,
                 epsilon: list,
                 exploration_stop: int,
                 observance_stop : int,
                 input_channel: int,
                 learning_rate: float,
                 target_update: int,
                 device: str
                 ):

        self.env = env
        self.temp_frames = deque(maxlen=input_channel)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon_list = epsilon
        self.input_channel = input_channel
        self.exploration_stop = exploration_stop
        self.observance_stop = observance_stop

        self.device = device
        self.net = Q_net(input_channel, env.action_space.n).apply(initialize_weights).to(self.device)
        self.target_net = Q_net(input_channel, env.action_space.n).apply(initialize_weights).to(self.device)
        self.Optimizer = torch.optim.RMSprop(self.net.parameters(), self.lr)
        self.updates = target_update


        self.frame_count = 0
        self.update_count = 0

    def take_action(self, state):

        if self.frame_count < self.observance_stop:
            # A uniform policy is run
            action = np.random.randint(self.env.action_space.n)
        else:
            # epsilon-greedy algorithm as Policy, annealed linearly
            decay = max(0, 1 - (self.frame_count-self.observance_stop) / (self.exploration_stop-self.observance_stop))
            epsilon = decay*(self.epsilon_list[0]-self.epsilon_list[1])+self.epsilon_list[1]

            # stack four into one
            self.temp_frames.append(state)

            if np.random.random() < epsilon:
                action = np.random.randint(self.env.action_space.n)
            else:
                # four frames are easy to gather, no need to consider repeat
                with torch.no_grad():
                    stacked_state = (torch.stack(list(self.temp_frames), dim=1)/255).to(self.device)  # stack recent 4
                    action = self.net(stacked_state).argmax().item()

        return action

    def update(self, replays):
        state, action, reward, next_state, done = replays
        state = (state/255).to(self.device)
        next_state = (next_state/255).to(self.device)
        action = action.long().view(-1,1).to(self.device)
        reward = reward.view(-1,1).to(self.device)
        done = done.view(-1,1).to(self.device)

        # a separate network in the Q-learning update.
        Q_values = self.net(state).gather(1, action)
        with torch.no_grad():
            Q_next_values = self.target_net(next_state).max(dim=1).values.view(-1,1)
            Q_target = reward + self.gamma * Q_next_values * (1 - done)
        DQN_loss = F.huber_loss(Q_values, Q_target)

        self.Optimizer.zero_grad()
        DQN_loss.backward()
        self.Optimizer.step()

        if self.update_count % self.updates == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        self.update_count += 1