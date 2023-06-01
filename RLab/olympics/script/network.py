import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class CNN_LSTM_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CNN_LSTM_Encoder, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(4, 2),
            nn.Conv2d(8, 8, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(4, 2),
            nn.Flatten(),
            )
        self.linear_in = nn.Linear(8 * 8 * 8, hidden_size)
        self.lstm = nn.LSTM(input_size, hidden_size)

    def forward(self, x):
        x = x.unsqueeze(0)# 将时间步维度转置到第一维
        x = x.transpose(0, 1)
        x = self.cnn_layers(x)
        _, (hidden, _) = self.lstm(x)
        return hidden.squeeze(0)


device = 'cpu'

class Actor(nn.Module):
    def __init__(self, state_space, action_space, hidden_size=64, cnn=True):
        super(Actor, self).__init__()
        self.is_cnn = cnn
        if self.is_cnn:
            self.encoder = CNN_LSTM_Encoder(input_size, hidden_size).to(device)
            state_space = hidden_size
        self.linear_in = nn.Linear(state_space, hidden_size)
        self.action_head = nn.Linear(hidden_size, action_space)

    def forward(self, x):
        if self.is_cnn:
            x = self.encoder(x)
        x = x.unsqueeze(0)
        x = F.relu(self.linear_in(x))
        action_prob = F.softmax(self.action_head(x),dim=1)
        return action_prob



class Critic(nn.Module):
    def __init__(self, state_space, hidden_size=64, cnn=True):
        super(Critic, self).__init__()
        self.is_cnn = cnn
        if self.is_cnn:
            self.encoder = CNN_LSTM_Encoder(input_size, hidden_size).to(device)
            state_space = hidden_size
        self.linear_in = nn.Linear(state_space, hidden_size)
        self.state_value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        if self.is_cnn:
            x = self.encoder(x)
        x = F.relu(self.linear_in(x))
        value = self.state_value(x)
        return value


class CNN_LSTM_Actor(nn.Module):
    def __init__(self, state_space, action_space, hidden_size=64):
        super(CNN_LSTM_Actor, self).__init__()
        self.encoder = CNN_LSTM_Encoder(input_size, hidden_size)
        self.linear_in = nn.Linear(hidden_size, hidden_size)
        self.action_head = nn.Linear(hidden_size, action_space)

    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(self.linear_in(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class CNN_LSTM_Critic(nn.Module):
    def __init__(self, state_space, hidden_size=64):
        super(CNN_LSTM_Critic, self).__init__()
        self.encoder = CNN_LSTM_Encoder(input_size, hidden_size)
        self.linear_in = nn.Linear(hidden_size, hidden_size)
        self.state_value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(self.linear_in(x))
        value = self.state_value(x)
        return value


class CategoricalActor(nn.Module):
    def __init__(self, state_space, action_space, hidden_size=64):
        super(CategoricalActor, self).__init__()
        self.encoder = CNN_LSTM_Encoder(input_size, hidden_size)
        self.linear_in = nn.Linear(hidden_size, hidden_size)
        self.action_head = nn.Linear(hidden_size, action_space)

    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(self.linear_in(x))
        action_logits = self.action_head(x)
        action_prob = F.softmax(action_logits, dim=1)
        return action_prob, action_logits


input_size = 512  # 设置输入大小
hidden_size = 64  # 设置隐藏层大小
action_space = 10  # 设置动作空间大小

# 创建 Actor 对象时传递正确的 action_space 参数
encoder = CNN_LSTM_Encoder(input_size, hidden_size).to(device)

actor = Actor(input_size, action_space, hidden_size).to(device)
critic = Critic(input_size, hidden_size).to(device)

cnn_lstm_actor = CNN_LSTM_Actor(input_size, action_space, hidden_size).to(device)
cnn_lstm_critic = CNN_LSTM_Critic(input_size, hidden_size).to(device)

categorical_actor = CategoricalActor(input_size, action_space, hidden_size).to(device)