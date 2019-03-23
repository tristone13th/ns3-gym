#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import argparse
from ns3gym import ns3env
from tcp_base import TcpTimeBased
from tcp_newreno import TcpNewReno
from tcp_marchons import TcpMarchons
from learner import Learner
import numpy as np

__author__ = "Piotr Gawlowicz"
__copyright__ = "Copyright (c) 2018, Technische Universität Berlin"
__version__ = "0.1.0"
__email__ = "gawlowicz@tkn.tu-berlin.de"


parser = argparse.ArgumentParser(description='Start simulation script on/off')
parser.add_argument('--start',
                    type=int,
                    default=1,
                    help='Start ns-3 simulation script 0/1, Default: 1')
parser.add_argument('--iterations',
                    type=int,
                    default=10,
                    help='Number of iterations, Default: 1')
args = parser.parse_args()

# 是否在开始时就启动脚本
startSim = bool(args.start)

# 迭代的次数
iterationNum = int(args.iterations)

port = 5555
simTime = 10  # seconds
stepTime = 0.5  # seconds
seed = 12
simArgs = {"--duration": simTime, }
debug = False

env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim,
                    simSeed=seed, simArgs=simArgs, debug=debug)
# simpler:
# env = ns3env.Ns3Env()
env.reset()

ob_space = env.observation_space
ac_space = env.action_space



print("Observation space: ", ob_space,  ob_space.dtype)
print("Action space: ", ac_space, ac_space.dtype)

stepIdx = 0
currIt = 0

# 每进行一次学习所需要的步数
learn_step = 5

# 每次学习的item数
learn_batch = 3

tcpAgent = TcpMarchons()


# initialize variable

list_of_action = []
list_of_action.append([1000])
list_of_action.append([700])
list_of_action.append([600])
list_of_action.append([550])
list_of_action.append([510])
list_of_action.append([0])
list_of_action.append([300])
list_of_action.append([400])
list_of_action.append([450])
list_of_action.append([490])
list_of_action.append([500])

# 初始化学习器
learner = Learner(list_of_action, in_dim = 4, h1 = 5, h2 = 5, out_dim = len(list_of_action))

try:
    while True:
        print("Start iteration: ", currIt)
        obs = env.reset()
        reward = 0
        done = False
        info = None
        cur_obs = torch.tensor([obs[5], obs[6], obs[7], obs[8]], dtype = torch.float)
        while True:
            stepIdx += 1
            pre_obs = cur_obs
            action = tcpAgent.get_action(obs, learner)
            print("---action: ", action)
            print("Step: ", stepIdx)

            # the num of env.step() be called equals to the sum of increasewindow and getssthresh be called
            obs, reward, done, info = env.step(action)
            # print("---obs, reward, done, info: ", obs, reward, done, info)
            cur_obs = torch.tensor([obs[5], obs[6], obs[7], obs[8]], dtype = torch.float)
            
            # 加入到库中，每一个item表示一个监督学习项
            learner.batch.add([pre_obs, action, reward, cur_obs])
            if stepIdx % learn_step == 0:
                learner.learnByLoss(learn_batch)
            

            if done:
                stepIdx = 0
                if currIt + 1 < iterationNum:
                    env.reset()
                break

        currIt += 1
        if currIt == iterationNum:
            break

except KeyboardInterrupt:
    print("Ctrl-C -> Exit")
finally:
    env.close()
    print("Done")
