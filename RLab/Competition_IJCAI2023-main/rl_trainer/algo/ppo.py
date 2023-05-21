from itertools import count
import os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import sys
from os import path
father_path = path.dirname(__file__)
sys.path.append(str(os.path.dirname(father_path)))
from rl_trainer.algo.network import Actor, Critic, CNN_Actor, CNN_Critic
from collections import namedtuple
from torch.utils.tensorboard import SummaryWriter
import datetime

class Args:
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 10
    buffer_capacity = 1000
    batch_size = 32
    gamma = 0.99
    lr = 0.0001

    weight_decay_actor = 1e-4
    weight_decay_critic = 1e-4
    #Trick 6 Learning Rate Decay

    action_space = 36
    # action_space = 3
    state_space = 1600
    use_grad_clip=True

args = Args()
device = 'cpu'

class PPO:
    clip_param = args.clip_param
    max_grad_norm = args.max_grad_norm
    ppo_update_time = args.ppo_update_time
    buffer_capacity = args.buffer_capacity
    batch_size = args.batch_size
    gamma = args.gamma
    action_space = args.action_space
    state_space = args.state_space
    lr = args.lr
    use_cnn = True
    weight_decay_actor = args.weight_decay_actor
    weight_decay_critic = args.weight_decay_critic
    use_grad_clip = args.use_grad_clip

    def __init__(self, run_dir=None):
        super(PPO, self).__init__()
        self.args = args
        if self.use_cnn:
            self.actor_net = CNN_Actor(self.state_space, self.action_space)
            self.critic_net = CNN_Critic(self.state_space)
        else:
            self.actor_net = Actor(self.state_space, self.action_space).to(device)
            self.critic_net = Critic(self.state_space).to(device)
        self.buffer = []
        self.counter = 0
        self.training_step = 0

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 
                lr=self.lr,weight_decay=self.weight_decay_actor,eps=1e-5)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 
                lr=self.lr,weight_decay=self.weight_decay_critic,eps=1e-5)

        if run_dir is not None:
            self.writer = SummaryWriter(os.path.join(run_dir, "PPO training loss at {}".format(
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))))
            #tensorboard
        self.IO = True if (run_dir is not None) else False

    def select_action(self, state, train=True):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action_prob = self.actor_net(state).to(device)
        c = Categorical(action_prob)
        if train:
            action = c.sample()
        else:
            action = torch.argmax(action_prob)
            # action = c.sample()
        return action.item(), action_prob[:, action.item()].item()

    def get_value(self, state):
        #获取V函数值，即critic网络的输出
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def store_transition(self, transition):
        #存储转移函数至缓存区
        #['state', 'action', 'a_log_prob', 'reward', 'next_state', 'done']
        #五元组
        self.buffer.append(transition)
        self.counter += 1

    def update(self, i_ep):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float).to(device)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1).to(device)
        reward = [t.reward for t in self.buffer]
        # update: don't need next_state
        # reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        # next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1).to(device)

        R = 0
        Gt = []
        for r in reward[::-1]:
            #[::-1] 倒序
            R = r + self.gamma * R
            Gt.insert(0, R)
            #insert(idx,obj) 在idx处加入obj. idx=0,相当于从头部开始加元素
        # Gt = rt + y rt+1 + y^2 rt+2 +...

        Gt = torch.tensor(Gt, dtype=torch.float).to(device)
        # print("The agent is updateing....")
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                # if self.training_step % 1000 == 0:
                #     print('I_ep {} ，train {} times'.format(i_ep, self.training_step))
                # with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1)
                #Tensor.view 调整tensor的维度. -1代表自动调整

                V = self.critic_net(state[index].squeeze(1))
                delta = Gt_index - V
                advantage = delta.detach()
                #使这个东西不具有梯度?
                
                # n-step advantage(GAE) = y^l*r(t+l).sum(l=0)-V(s t)


                # epoch iteration, PPO core!!!
                action_prob = self.actor_net(state[index].squeeze(1)).gather(1, action[index])  # new policy

                ratio = (action_prob / old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # y =mix(max(x,min_v),max_v) 
                
                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                # self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                if self.use_grad_clip: # Trick 7: Gradient clip  
                    nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # update critic network
                value_loss = F.mse_loss(Gt_index, V)
                # self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                if self.use_grad_clip: # Trick 7: Gradient clip 
                    nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

                if self.IO:
                    self.writer.add_scalar('loss/policy loss', action_loss.item(), self.training_step)
                    self.writer.add_scalar('loss/critic loss', value_loss.item(), self.training_step)

        # del self.buffer[:]  # clear experience
        self.clear_buffer()


    def clear_buffer(self):
        #清空缓存区
        del self.buffer[:]

    def save(self, save_path, episode):
        #保存模型
        base_path = os.path.join(save_path, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_actor_path = os.path.join(base_path, "actor_" + str(episode) + ".pth")
        torch.save(self.actor_net.state_dict(), model_actor_path)
        model_critic_path = os.path.join(base_path, "critic_" + str(episode) + ".pth")
        torch.save(self.critic_net.state_dict(), model_critic_path)

    def load(self, run_dir, episode):
        #载入模型
        print(f'\nBegin to load model: ')
        print("run_dir: ", run_dir)
        base_path = os.path.dirname(os.path.dirname(__file__))
        print("base_path: ", base_path)
        algo_path = os.path.join(base_path, 'models/ppo')
        run_path = os.path.join(algo_path, run_dir)
        run_path = os.path.join(run_path, 'trained_model')
        model_actor_path = os.path.join(run_path, "actor_" + str(episode) + ".pth")
        model_critic_path = os.path.join(run_path, "critic_" + str(episode) + ".pth")
        print(f'Actor path: {model_actor_path}')
        print(f'Critic path: {model_critic_path}')

        if os.path.exists(model_critic_path) and os.path.exists(model_actor_path):
            actor = torch.load(model_actor_path, map_location=device)
            critic = torch.load(model_critic_path, map_location=device)
            self.actor_net.load_state_dict(actor)
            self.critic_net.load_state_dict(critic)
            print("Model loaded!")
        else:
            sys.exit(f'Model not founded!')

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now



