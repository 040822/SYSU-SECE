import numpy as np


class RunningMeanStd:
    # Dynamically calculate mean and std
    # 动态计算平均值和标准差
    def __init__(self, shape):  # shape:the dimension of input data，即state的shape
        self.n = 0
        self.mean = np.zeros(shape) #平均值
        self.S = np.zeros(shape)    #方差
        self.std = np.sqrt(self.S)  #标准差

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
        #核心思想：已知n个数的mean和std，求增加一个数后的mean和std
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)
        # Normalize 传入的state
        return x


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape) #算reword的std
        self.R = np.zeros(self.shape) #R实际上是一个一维值 (shape=1)

    def __call__(self, x):
        self.R = self.gamma * self.R + x #类似于梯度下降？
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std 除以std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape) #重置计算后的Reward(self.R)
