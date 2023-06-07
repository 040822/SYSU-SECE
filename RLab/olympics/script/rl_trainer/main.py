# here lies an example of how to train an RL agent


import argparse
import datetime

from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import os
from pathlib import Path
import sys
base_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_dir)
engine_path = os.path.join(base_dir, "olympics_engine")
sys.path.append(engine_path)


from collections import deque, namedtuple


from env.chooseenv import make
from rl_trainer.log_path import *
from rl_trainer.algo.ppo import PPO
from rl_trainer.algo.normalization import Normalization,RewardScaling
from rl_trainer.algo.random import random_agent


parser = argparse.ArgumentParser()
parser.add_argument('--game_name', default="olympics-integrated", type=str)
parser.add_argument('--algo', default="ppo", type=str, help="ppo/sac")
parser.add_argument('--max_episodes', default=1500, type=int)    #1500
parser.add_argument('--episode_length', default=5, type=int)   #500
parser.add_argument('--map', default=1, type = int)

parser.add_argument('--seed', default=1, type=int)

parser.add_argument("--save_interval", default=100, type=int)
parser.add_argument("--model_episode", default=0, type=int)

parser.add_argument("--use_model",default=0) #1 开始时加载模型， 0 不加载开新模型
parser.add_argument("--load_model", default=0, action='store_true') #0是训练，1是不训
parser.add_argument("--load_run", default=14, type=int)
parser.add_argument("--load_episode", default=1500, type=int)

# PPO优化中用到的参数
parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
parser.add_argument("--reward_gamma", type=float, default=0.99, help="Discount factor")


device = 'cpu'
RENDER = False

speed = [-100,-40,20,60,80,140,200]
angle = [-30,-20,-10,-5, 0, 5, 10, 20, 30]
actions_map=[[i,j] for i in speed for j in angle]

def main(args):
    print("==algo: ", args.algo)
    print(f'device: {device}')
    print(f'model episode: {args.model_episode}')
    print(f'save interval: {args.save_interval}')

    env = make(args.game_name)          #build environment


    num_agents = env.n_player
    print(f'Total agent number: {num_agents}')

    ctrl_agent_index = 1
    print(f'Agent control by the actor: {ctrl_agent_index}')

    ctrl_agent_num = 1

    width = env.env_core.view_setting['width']+2*env.env_core.view_setting['edge']
    height = env.env_core.view_setting['height']+2*env.env_core.view_setting['edge']
    print(f'Game board width: {width}')
    print(f'Game board height: {height}')

    act_dim = env.action_dim
    obs_dim = 40*40
    print(f'action dimension: {act_dim}')
    print(f'observation dimension: {obs_dim}')

    torch.manual_seed(args.seed)
    # define checkpoint path
    run_dir, log_dir = make_logpath(args.game_name, args.algo)
    if not args.load_model:
        writer = SummaryWriter(os.path.join(str(log_dir), "{}_{} on olympics-integrated".format(
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),args.algo)))
        save_config(args, log_dir)

    record_win = deque(maxlen=100)
    record_win_op = deque(maxlen=100)

    if args.load_model:         #setup algos
        model = PPO(max_train_steps=args.max_episodes)
        load_dir = os.path.join(os.path.dirname(run_dir), "run" + str(args.load_run))
        model.load(load_dir,episode=args.load_episode)
    else:
        model = PPO(run_dir,max_train_steps=args.max_episodes)
    
    Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state', 'done'])

    opponent_agent = random_agent()     #we use random opponent agent here

    episode = 0
    train_count = 0
    state_norm = Normalization(shape=[40,40])  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.reward_gamma)


    
    while episode < args.max_episodes:
        env = make(args.game_name)          #rebuild each time to shuffle the running map
        state = env.reset()
        
        if RENDER:
            env.env_core.render()
        # obs_ctrl_agent = np.array(state[ctrl_agent_index]['obs']['agent_obs']).flatten()
        obs_ctrl_agent = np.array(state[ctrl_agent_index]['obs']['agent_obs'])
        NEW_GAME_flag = state[ctrl_agent_index]['obs']['game_mode']
        obs_oppo_agent = np.array(state[1-ctrl_agent_index]['obs']['agent_obs'])   #[25,25]

        if args.use_reward_scaling:
            reward_scaling.reset() #每一个episode初始化一次
        if args.use_state_norm:
            obs_ctrl_agent = state_norm(obs_ctrl_agent)# Trick 2:state normalization

        episode += 1
        step = 0
        Gt = 0

        while True:
            action_opponent = opponent_agent.act(obs_oppo_agent)        #opponent action
            action_opponent = [[0],[0]]  #here we assume the opponent is not moving in the demo

            action_ctrl_raw, action_prob= model.select_action(obs_ctrl_agent, False if args.load_model else True)
                            #inference
            action_ctrl = actions_map[action_ctrl_raw]
            action_ctrl = [[action_ctrl[0]], [action_ctrl[1]]]        #wrapping up the action
            #print(action_ctrl)
            action = [action_opponent, action_ctrl] if ctrl_agent_index == 1 else [action_ctrl, action_opponent]

            next_state, reward, done, _, info = env.step(action)

            # next_obs_ctrl_agent = np.array(next_state[ctrl_agent_index]['obs']['agent_obs']).flatten()
            next_obs_ctrl_agent = np.array(next_state[ctrl_agent_index]['obs']['agent_obs'])
            next_obs_oppo_agent = next_state[1-ctrl_agent_index]['obs']

            if args.use_reward_norm:
                reward = reward_norm(reward)
            elif args.use_reward_scaling:
                reward = reward_scaling(reward)
            if args.use_state_norm:
                next_obs_ctrl_agent = state_norm(next_obs_ctrl_agent)# Trick 2:state normalization

            step += 1

            if not done:            #reward shaping, here we simply penality every time step
                post_reward = [-1., -1.]
            else:
                if reward[0] != reward[1]:
                    post_reward = [reward[0]-100, reward[1]] if reward[0]<reward[1] else [reward[0], reward[1]-100]
                else:
                    post_reward=[-1., -1.]

            if not args.load_model:
                trans = Transition(obs_ctrl_agent, action_ctrl_raw, action_prob, post_reward[ctrl_agent_index],
                                   next_obs_ctrl_agent, done)
                model.store_transition(trans)


            obs_oppo_agent = next_obs_oppo_agent
            obs_ctrl_agent = next_obs_ctrl_agent
            if RENDER:
                env.env_core.render()
            Gt += reward[ctrl_agent_index] if done else -1

            if done:
                win_is = 1 if reward[ctrl_agent_index]>reward[1-ctrl_agent_index] else 0
                win_is_op = 1 if reward[ctrl_agent_index]<reward[1-ctrl_agent_index] else 0
                record_win.append(win_is)
                record_win_op.append(win_is_op)
                print("Episode: ", episode, "controlled agent: ", ctrl_agent_index, "; Episode Return: ", Gt,
                      "; win rate(controlled & opponent): ", '%.2f' % (sum(record_win)/len(record_win)),
                      '%.2f' % (sum(record_win_op)/len(record_win_op)), '; Trained episode:', train_count)

                if not args.load_model:
                    if args.algo == 'ppo' and len(model.buffer) >= model.batch_size:
                        
                        #model.update(episode)           #model training
                        if win_is == 1:
                            model.update(episode)           #model training
                            train_count += 1
                        else:
                            model.clear_buffer()

                    writer.add_scalar('training Gt', Gt, episode)

                break
        if episode % args.save_interval == 0 and not args.load_model:
            model.save(run_dir, episode)





if __name__ == '__main__':
    args = parser.parse_args()
    #args.load_model = True
    #args.load_run = 3
    #args.map = 3
    #args.load_episode= 900
    main(args)
