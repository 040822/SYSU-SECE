import sys
sys.path.append("Competition_IJCAI2023-main")
from pathlib import Path
base_path = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_path)
print(sys.path)
from olympics_engine.generator import create_scenario
import argparse
from olympics_engine.agent import *
import time

from scenario import Running, table_hockey, football, wrestling, billiard, \
    curling, billiard_joint,curling_joint, curling_long, curling_competition, Running_competition, billiard_competition, Seeks

from AI_olympics import AI_Olympics

import random
import json



def store(record, name):

    with open('logs/'+name+'.json', 'w') as f:
        f.write(json.dumps(record))

def load_record(path):
    file = open(path, "rb")
    filejson = json.load(file)
    return filejson

RENDER = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', default="all_v2", type= str,
                        help = 'running/table-hockey/football/wrestling/billiard/curling/all/all_v2')           
    parser.add_argument("--seed", default=1, type=int)
    args = parser.parse_args()
    #argsparse是python的命令行解析的标准模块，内置于python，不需要安装。
    #这个库可以让我们直接在命令行中就可以向程序中传入参数并让程序运行。
    #type是要传入的参数的数据类型  help是该参数的提示信息

    for i in range(1):
        if 'all' not in args.map:
            Gamemap = create_scenario(args.map)
            # generator.py/create_scenario
        #game = table_hockey(Gamemap)
        if args.map == 'running':
            game = Running(Gamemap)
            agent_num = 2
        elif args.map == 'running-competition':

            map_id = random.randint(1,10)
            # map_id = 3
            Gamemap = create_scenario(args.map)
            game = Running_competition(meta_map=Gamemap,map_id=map_id)
            agent_num = 2

        elif args.map == 'seeks':
            game = Seeks(Gamemap)
            agent_num=2

        elif args.map == 'table-hockey':
            game = table_hockey(Gamemap)
            agent_num = 2
        elif args.map == 'football':
            game = football(Gamemap)
            agent_num = 2
        elif args.map == 'wrestling':
            game = wrestling(Gamemap)
            agent_num = 2
        # elif args.map == 'volleyball':
        #     game = volleyball(Gamemap)
        #     agent_num = 2
        elif args.map == 'billiard':
            game = billiard(Gamemap)
            agent_num = 2
        elif args.map == 'billiard-competition':
            game = billiard_competition(Gamemap)
            agent_num = 2

        elif args.map == 'curling':
            game = curling(Gamemap)
            agent_num = 2

        elif args.map == 'curling-joint':
            game = curling_joint(Gamemap)
            agent_num = 2

        elif args.map == 'billiard-joint':
            game = billiard_joint(Gamemap)
            agent_num = 2

        elif args.map == 'curling-long':
            game = curling_long(Gamemap)
            agent_num = 2

        elif args.map == 'curling-competition':
            game = curling_competition(Gamemap)
            agent_num = 2

        elif args.map == 'all':
            game = AI_Olympics(random_selection = False, minimap=False)
            agent_num = 2

        elif args.map == 'all_v2':
            game = AI_Olympics(random_selection=False, minimap=False, vis=300, vis_clear=5)
            agent_num = 2

        agent = random_agent()
        rand_agent = random_agent()

        obs = game.reset()
        done = False
        step = 0
        if RENDER:
            game.render()

        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        time_epi_s = time.time()
        while not done:
            step += 1

            # print('\n Step ', step)

            #action1 = [100,0]#agent.act(obs)
            #action2 = [100,0] #rand_agent.act(obs)
            if agent_num == 2:
                action1, action2 = agent.act(obs[0]), rand_agent.act(obs[1])
                # action1 = [50,0.1]
                # action2 = [100,-0.2]

                # action1 =[50,1]
                # action2 = [50,-1]


                action = [action1, action2]
            elif agent_num == 1:
                action1 = agent.act(obs)
                action = [action1]

            # if step <= 5:
            #     action = [[200,0]]
            # else:
            #     action = [[0,0]]
            # action = [[200,action1[1]]]

            obs, reward, done, _ = game.step(action)
            print(f'reward = {reward}')
            # print('obs = ', obs)
            # plt.imshow(obs[0])
            # plt.show()
            if RENDER:
                game.render()

        duration_t = time.time() - time_epi_s
        print("episode duration: ", duration_t,
              "step: ", step,
              "time-per-step:",(duration_t)/step)
        # if args.map == 'billiard':
        #     print('reward =', game.total_reward)
        # else:
            # print('reward = ', reward)
        # if R:
        #     store(record,'bug1')

