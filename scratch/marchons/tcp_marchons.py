__author__ = "Tristone13th"
__copyright__ = "Copyright (c) 2019, Tristone13th"
__version__ = "0.0.0"
__email__ = "tristone13th@outlook.com"

from tcp_base import TcpEventBased
import torch
from learner import Learner
import numpy as np


# This class is designed for using reinforcement learning in congestion control
# obs: box(16)
# reward: float
# done: bool
# info: string
class TcpMarchons(TcpEventBased):
    """docstring for TcpMarchons"""
    def __init__(self):
        super(TcpMarchons, self).__init__()
        print("You are using TcpMarchons to control network congestion which is a new designed reinforcement learning-based algorithm.")

    # 记录上一个时刻的状态和动作
    # 这个函数的作用是根据当前准哪个台
    def get_action(self, obs, learner):

        # 当前的拥塞窗口
        cWnd = obs[5]
        # 段大小
        segmentSize = obs[6]
        # 已经被应答的段

        segmentsAcked = obs[7]
        # 仍旧在传输中的段
        bytesInFlight  = obs[8]

        # 初始化新的拥塞窗口
        action = 1
        state = torch.tensor([cWnd, segmentSize, segmentsAcked, bytesInFlight], dtype = torch.float)
        print("cur state:", state)

        action = learner.getAction(state)

        return action