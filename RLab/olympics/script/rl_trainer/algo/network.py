import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import traceback

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

class CNN_encoder(nn.Module):
    def __init__(self):
        super(CNN_encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(4, 2),
            nn.Conv2d(8, 8, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(4,2),
            nn.Flatten()
        )

    def forward(self, view_state):
        # [batch, 128]
        #view_state = view_state.unsqueeze(1)  # 在第1维上扩展一维，表示channel为1
        x = self.net(view_state)
        return x

#device = 'cuda'
device = 'cpu'

class Actor(nn.Module):
    def __init__(self, state_space, action_space, hidden_size=64, cnn=True):
        super(Actor, self).__init__()
        self.is_cnn = cnn
        if self.is_cnn:
            self.encoder = CNN_encoder().to(device)
            state_space = 512
        self.linear_in = nn.Linear(state_space, hidden_size)
        # 512=>64
        self.action_head = nn.Linear(hidden_size, action_space)
        # 64=>action_space

    def forward(self, x):
        x = x.unsqueeze(1)  # 在第1维上扩展一维，表示channel为1
        if self.is_cnn:
            x = self.encoder(x)
        x = F.relu(self.linear_in(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        #action_prob = F.softmax(self.action_head(x), dim=-1)
        return action_prob


class Critic(nn.Module):
    def __init__(self, state_space, hidden_size=64, cnn=True):
        super(Critic, self).__init__()
        self.is_cnn = cnn
        if self.is_cnn:
            self.encoder = CNN_encoder().to(device)  # 用GPU计算
            state_space = 512
        self.linear_in = nn.Linear(state_space, hidden_size)
        # 512=>64
        self.state_value = nn.Linear(hidden_size, 1)
        # 64=>1

    def forward(self, x,batch_size):
        #x = x.unsqueeze(1)  # 在第1维上扩展一维，表示channel为1 会出现[40,1,40]的情况,故弃用.
        x=x.reshape(batch_size,1,40,40)  #batch_size代表输入的帧数
        if self.is_cnn:
            x = self.encoder(x)
        x = F.relu(self.linear_in(x))
        value = self.state_value(x)
        return value
        #[2602,1]











