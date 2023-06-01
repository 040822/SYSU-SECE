import argparse
import os
from pathlib import Path
import sys
import torch
import numpy as np
base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))

print("hello?")



actions_map = {0: [-100, -30], 1: [-100, -18], 2: [-100, -6], 3: [-100, 6], 4: [-100, 18], 5: [-100, 30], 6: [-40, -30],
               7: [-40, -18], 8: [-40, -6], 9: [-40, 6], 10: [-40, 18], 11: [-40, 30], 12: [20, -30], 13: [20, -18],
               14: [20, -6], 15: [20, 6], 16: [20, 18], 17: [20, 30], 18: [80, -30], 19: [80, -18], 20: [80, -6],
               21: [80, 6], 22: [80, 18], 23: [80, 30], 24: [140, -30], 25: [140, -18], 26: [140, -6], 27: [140, 6],
               28: [140, 18], 29: [140, 30], 30: [200, -30], 31: [200, -18], 32: [200, -6], 33: [200, 6], 34: [200, 18],
               35: [200, 30]}           #dicretise action space
import ppo
from ppo import PPO
model = PPO()
model.loadX(load_dir=os.path.dirname(os.path.abspath(__file__)),actorname="actor_10000",criticname="critic_10100")


sys.path.pop(-1)  # just for safety

def my_controller(obs_list,action_space_list,obs_space_list):
    observation_copy = obs_list.copy()
    observation = observation_copy["obs"]
    agent_id = observation_copy["controlled_player_index"]
    #see =observation['agent_obs']
    #print(">>>> ",see)
    action_ctrl_raw, action_prob= model.select_action(np.array(observation) , False)
    action_ctrl = actions_map[action_ctrl_raw]
    action_ctrl = [[action_ctrl[0]], [action_ctrl[1]]]        #wrapping up the action
    result = [np.array(x, dtype=np.float32) for x in action_ctrl]
    return result
