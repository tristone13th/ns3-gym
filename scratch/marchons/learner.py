__author__ = "tristone13th"
__copyright__ = "tristone13th"
__email__ = "tristone13th@outlook.com"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# 默认为系统时间
np.random.seed()



# This class is designed for implementing batch
class Batch(object):
    def __init__(self, dataset = None, capacity = 100):
        super(Batch, self).__init__()
        self.__capacity = capacity
        if dataset == None:
            self.__dataset = []
        else:
            assert len(dataset) <= capacity
            self.__dataset = dataset

    def sample(self, sample_size):
        assert len(self.__dataset) >= sample_size
        return random.sample(self.__dataset, sample_size)

    def add(self, element):
        if(len(self.__dataset) < self.__capacity):
            self.__dataset.append(element)
        else:
            self.__dataset.pop(0)
            self.__dataset.append(element)
        
    def getSize(self):
        return len(self.__dataset)


# This class is the reinforcement learning learner
class Learner(object):
    def __init__(self, list_of_action, in_dim, h1, h2, out_dim, alpha=0.5, gamma=0.5, epsilon=0.2, pre_dataset = None, capacity = 100):
        super(Learner, self).__init__()
        self.batch = Batch(pre_dataset, capacity)
        self.__net = Net(in_dim, h1, h2, out_dim)
        self.initNetPara()
        
        # 将Action转化为对于index的映射
        self.__action_to_index = {}
        self.__list_of_action = list_of_action
        self.__action_num = len(list_of_action)
        self.buildActionIdx()

        # 元参数
        self.__alpha = alpha
        self.__gamma = gamma
        self.__epsilon = epsilon


    # 初始化网络参数
    def initNetPara(self):
        for name, para in self.__net.named_parameters():
            para = np.random.random()

    # 决策函数,利用和探索
    def getAction(self, state):
        if np.random.uniform(0, 1) < self.__epsilon:
            randint = np.random.randint(0, self.__action_num)
            print("suiji")
            return self.__list_of_action[randint]
        else:
            output = self.__net(state)
            maxi = 0
            maxv = float("-inf")
            for i in range(len(output)):
                if output[i] > maxv:
                    maxi = i
                    maxv = output[i]
            return self.__list_of_action[i]
        
    # 批量进行学习
    def learnByLoss(self, learn_batch):
        print("before-------------------------------------")
        print(self.__net.parameters)
        dataset = self.batch.sample(learn_batch)
        for i in range(learn_batch):
            pre_state = dataset[i][0]
            action = dataset[i][1]
            action_index = self.__action_to_index[self.action2String(action)]
            reward = dataset[i][2]
            cur_state = dataset[i][3]
            q_s_a = self.__net(pre_state)[action_index]
            q_news_newa = max(self.__net(cur_state))
            loss_without_squre = reward + self.__gamma * q_news_newa - q_s_a
            loss = loss_without_squre * loss_without_squre
            loss.backward()
        print("after-------------------------------------")
        print(self.__net.parameters)



    # 将动作转化为字符串
    def action2String(self, action):
        final_str = ""
        isStart = True
        for item in action:
            if isStart:
                final_str = final_str + str(item)
                isStart = False
            else:
                final_str = final_str + ' ' + str(item)
        return final_str



    # 使用SARSA算法进行学习

    # 初始化建立关于所有Action的键值对
    def buildActionIdx(self):
        for i in range(self.__action_num):
            key = self.action2String(self.__list_of_action[i])
            self.__action_to_index.update({key: i})
        

    # def learnWithSarsa(self, pre_state, pre_action, reward, state, action):
    #     metocarlo_loss = self.__alpha * \
    #         (reward + self.__gamma * self.getQValue(state, action) -
    #          self.getQValue(pre_state, pre_action))
    #     new_value = self.getQValue(pre_state, pre_action) + metocarlo_loss
    #     self.updateQValue(pre_state, pre_action, new_value)




    # 得到q函数值，如果没有则增加一个默认值
    def getQValue(self, state, action):
        return self.__net(state)


   

class Net(nn.Module):
    # 构造函数
    def __init__(self, in_dim = 5, n_hidden_1 = 10, n_hidden_2 = 10, out_dim = 5):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.sigmoid1(x)
        x = self.layer2(x)
        x = self.sigmoid2(x)
        x = self.layer3(x)
        return x
