# -*- coding: utf-8 -*-
# @Time    : 2022/4/6 17:52
# @Author  : 耀
# from collections import namedtuple
# from itertools import count

import os
import random

import numpy as np
import argparse  # argparse是python用于解析命令行参数和选项的标准模块。作用是用于解析命令行参数。
import pandas as pd
import time
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import threading
from socket import *
import TCP.TCPdata
from matplotlib import pyplot as plt

# from torch.distributions import Normal

# from tensorboardX import SummaryWriter


host = '127.0.0.1'
port = 8085
bufsiz = 2048 * 32  # 2048 * 16
addr = (host, port)

global left, right, get_tcp_state, reset


#  输入推力
def input_control():
    while True:
        global left
        global right
        temp = input('Left:')
        left = temp
        temp = input('Right:')
        right = temp


def TCPcommuition():
    # 创建tcp套接字，绑定，监听
    tcpServerSock = socket(AF_INET, SOCK_STREAM)  # 创建TCP Socket
    # AF_INET 服务器之间网络通信
    # socket.SOCK_STREAM 流式socket , for TCP
    tcpServerSock.bind(addr)  # 将套接字绑定到地址,
    # 在AF_INET下,以元组（host,port）的形式表示地址.
    tcpServerSock.listen(5)  # 操作系统可以挂起的最大连接数量，至少为1，大部分为5

    while True:
        global left, right, get_tcp_state, reset
        reset = 0
        print('waiting for connection')
        # tcp这里接收到的是客户端的sock对象，后面接受数据时使用socket.recv()
        tcpClientSock, addr2 = tcpServerSock.accept()  # 接受客户的连接
        # 接受TCP连接并返回（conn,address）,其中conn是新的套接字对象，
        # 可以用来接收和发送数据。
        # address是连接客户端的地址。
        print('connected from :', addr2)

        left = 0
        right = 0
        # t1 = threading.Thread(target=input_control, name='T1')  # 输入推力控制
        # t1.start()
        while True:
            data = tcpClientSock.recv(bufsiz)  # 接收客户端发来的数据
            if not data:
                break
            # 接收数据
            # time.sleep(0.01)
            # start = time.time()
            ReceveData = data.decode()
            distance_terminal, posture, distance, warning, block, overturn, reach, velocity, destination_angle = TCP.TCPdata.deal(ReceveData)
            get_tcp_state = [distance_terminal, posture, distance, warning, block, overturn, reach, velocity, destination_angle]
            # 发送数据

            # msg = str('123') + ',' + str(left) + ',' + str(right) + ',' + str(block) + ',' + str(overturn) + ',' + \
            #       str(reach) + ',' + str(reset) + ',' + str('321')  # 测试传入UE4控制量是否正常
            msg = str('123') + ',' + str(left) + ',' + str(right) + ',' + str(0) + ',' + str(0) + ',' + str(0) + ',' + str(reset) + ',' + str('321')
            tcpClientSock.send(msg.encode())  # 返回给客户端数据
            # end = time.time()
            # print(end - start)
        tcpClientSock.close()
    tcpServerSock.close()


t2 = threading.Thread(target=TCPcommuition, name='T2')
t2.start()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = argparse.ArgumentParser()

parser.add_argument('--mode', default='test', type=str)  # mode = 'train' or 'test'
parser.add_argument('--tau', default=0.005, type=float)  # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--iteration', default=5, type=int)

parser.add_argument('--learning_rate', default=3e-4, type=float)
parser.add_argument('--gamma', default=0.99, type=int)  # discounted factor
parser.add_argument('--capacity', default=200000, type=int)  # replay buffer size
parser.add_argument('--num_iteration', default=100000, type=int)  # num of  games
parser.add_argument('--batch_size', default=100, type=int)  # mini batch size
parser.add_argument('--seed', default=1, type=int)

# optional parameters
parser.add_argument('--num_hidden_layers', default=2, type=int)
parser.add_argument('--sample_frequency', default=256, type=int)
parser.add_argument('--activation', default='Relu', type=str)
parser.add_argument('--render', default=False, type=bool)  # show UI or not
parser.add_argument('--log_interval', default=50, type=int)  #
parser.add_argument('--load', default=False, type=bool)  # load model
parser.add_argument('--render_interval', default=100, type=int)  # after render_interval, the env.render() will work
parser.add_argument('--policy_noise', default=0.1, type=float)
parser.add_argument('--noise_clip', default=0.1, type=float)
parser.add_argument('--policy_delay', default=2, type=int)
parser.add_argument('--exploration_noise', default=45000, type=float)
parser.add_argument('--max_episode', default=2000, type=int)
parser.add_argument('--print_log', default=5, type=int)
args = parser.parse_args()

# Set seeds
# env.seed(args.seed)
# torch.manual_seed(args.seed)
# np.random.seed(args.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_name = os.path.basename(__file__)
# env = gym.make(args.env_name)

state_dim = 24  # 24维数据

action_dim = 2
max_action = 35000

min_Val = torch.tensor(0).float().to(device)  # min value

directory = './exp' + script_name + 'USV' + './'
'''
Implementation of TD3 with pytorch 
Original paper: https://arxiv.org/abs/1802.09477
Not the author's implementation !
'''


def Reward(difference_distance_terminal, distance_terminal, destination_angle, posture, warning, block, overturn, reach, velocity):
    global left, right
    restart = 0
    reward = 0
    # l1 = 0.2  # block_reward
    # l2 = 0  # overturn_reward
    # l3 = 0.3  # reach_reward
    # l4 = 0.1  # warning_reward
    # l5 = 0  # posture_reward
    # l6 = 0.1  # difference_distance_terminal_reward
    # l7 = 0.1  # destination_angle_reward
    # l8 = 0.2  # distance_terminal_reward
    block_reward, overturn_reward, reach_reward, warning_reward, velocity_reward, posture_reward, difference_distance_terminal_reward, distance_terminal_reward = 0, 0, 0, 0, 0, 0, 0, 0
    if block == 1:
        block_reward = -2
        restart = 1
    if overturn == 1:
        overturn_reward = -1
        restart = 1
    if reach == 1:
        reach_reward = 20
        restart = 1
    if warning == 1:
        warning_reward = -1
    if velocity < 200 / 1000:
        if -1 / ((velocity + 0.00001) * 10) < -2:
            velocity_reward = -2
    elif velocity >= 200 / 1000:
        velocity_reward = 1

    reward += -0.05
    distance_terminal_reward = 0.1 * (1 / distance_terminal)
    # posture_reward = - 0.1*((posture[0] - 0.5) + (posture[1] - 0.5))
    difference_distance_terminal_reward = difference_distance_terminal * 200  # 正常都是0.00X的数
    destination_angle_reward = - abs(destination_angle)*10
    # print('block_reward:{} overturn_reward:{} reach_reward:{} warning_reward:{} difference_distance_terminal_reward:{} distance_terminal_reward:{} destination_angle_reward:{} velocity_reward:{} reward:{}\n\n'
    #       .format(block_reward, overturn_reward, reach_reward, warning_reward, difference_distance_terminal_reward, distance_terminal_reward, destination_angle_reward, velocity_reward, reward))
    reward = block_reward + overturn_reward + reach_reward + warning_reward + difference_distance_terminal_reward + distance_terminal_reward + destination_angle_reward + velocity_reward + reward
    # print(reward)
    return reward, restart


def deal_main_data(get_tcp_state):
    get_state = get_tcp_state
    distance_terminal = get_state[0]
    posture = get_state[1]
    distance = get_state[2]
    warning = get_state[3]
    block = get_state[4]
    overturn = get_state[5]
    reach = get_state[6]
    velocity = get_state[7]
    destination_angle = get_state[8]

    return distance_terminal, posture, distance, warning, block, overturn, reach, velocity, destination_angle


def Next_state(action):
    global left
    global right
    left, right = int(action[0]), int(action[1])
    # print(action[0],action[1])
    time.sleep(0.5)
    distance_terminal, posture, distance, warning, block, overturn, reach, velocity, destination_angle = deal_main_data(
        get_tcp_state)
    nextstate = np.append(distance, posture)
    nextstate = np.append(nextstate, destination_angle)
    nextstate = np.append(nextstate, velocity)
    nextstate = np.append(nextstate, distance_terminal)
    return nextstate, distance_terminal, destination_angle, posture, warning, block, overturn, reach, velocity


class Replay_buffer():

    def __init__(self, max_size=args.capacity):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 400)
        # self.fc2=nn.LSTM(400,300)
        self.fc2 = nn.Linear(400, 256)
        self.fc3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a))
        a = torch.tanh(self.fc3(a)) * self.max_action
        return a


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)

        q = F.relu(self.fc1(state_action))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)
        return q


class TD3():
    def __init__(self, state_dim, action_dim, max_action):
        self.replay_buffer = Replay_buffer()
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_1_target = Critic(state_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_2_target = Critic(state_dim, action_dim).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=1e-3)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=1e-3)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.max_action = max_action
        self.memory = Replay_buffer(args.capacity)
        # self.writer = SummaryWriter(directory)
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):
        state = torch.tensor(state.reshape(1, -1)).float().to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self, num_iteration):

        if self.num_training % 10 == 0:
            print("====================================")
            print("已经训练{}次...".format(self.num_training))
            print("====================================")
        for i in range(num_iteration):
            x, y, u, r, d = self.memory.sample(args.batch_size)
            # print('x:{}\n y:{}\n u:{}\n r:{}\n d:{}\n'.format(x, y, u, r, d))
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Select next action according to target policy:
            noise = torch.ones_like(action).data.normal_(0, args.policy_noise).to(device)
            noise = noise.clamp(-args.noise_clip, args.noise_clip)

            next_action = (self.actor_target(next_state) + noise)
            next_action = next_action.clamp(-self.max_action, self.max_action)

            # Compute target Q-value:
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * args.gamma * target_Q).detach()

            # Optimize Critic 1:
            current_Q1 = self.critic_1(state, action)
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()
            # self.writer.add_scalar('Loss/Q1_loss', loss_Q1, global_step=self.num_critic_update_iteration)

            # Optimize Critic 2:
            current_Q2 = self.critic_2(state, action)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()
            # self.writer.add_scalar('Loss/Q2_loss', loss_Q2, global_step=self.num_critic_update_iteration)
            # Delayed policy updates:
            if i % args.policy_delay == 0:
                # Compute actor loss:
                actor_loss = - self.critic_1(state, self.actor(state)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                # self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(((1 - args.tau) * target_param.data) + args.tau * param.data)

                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_(((1 - args.tau) * target_param.data) + args.tau * param.data)

                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_(((1 - args.tau) * target_param.data) + args.tau * param.data)

                self.num_actor_update_iteration += 1
        self.num_critic_update_iteration += 1
        self.num_training += 1

    def save(self):
        if os.path.exists(directory) == 1:
            torch.save(self.actor.state_dict(), directory + 'actor.pth')
            torch.save(self.actor_target.state_dict(), directory + 'actor_target.pth')
            torch.save(self.critic_1.state_dict(), directory + 'critic_1.pth')
            torch.save(self.critic_1_target.state_dict(), directory + 'critic_1_target.pth')
            torch.save(self.critic_2.state_dict(), directory + 'critic_2.pth')
            torch.save(self.critic_2_target.state_dict(), directory + 'critic_2_target.pth')
        else:
            os.mkdir(directory)
        # print("====================================")
        # print("Model has been saved...")
        # print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
        self.actor_target.load_state_dict(torch.load(directory + 'actor_target.pth'))
        self.critic_1.load_state_dict(torch.load(directory + 'critic_1.pth'))
        self.critic_1_target.load_state_dict(torch.load(directory + 'critic_1_target.pth'))
        self.critic_2.load_state_dict(torch.load(directory + 'critic_2.pth'))
        self.critic_2_target.load_state_dict(torch.load(directory + 'critic_2_target.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")


def last_100_mean_reward(last_100_reward):
    sum = 0
    for item in last_100_reward:
        sum += item
    return sum / len(last_100_reward)


def main():
    global left, right, get_tcp_state, reset
    agent = TD3(state_dim, action_dim, max_action)
    print("开始")
    time.sleep(3)
    reset = 1
    if args.mode == 'train':
        # if args.load:
        #     agent.load()
        total_step = 0
        return_list = []
        mean_return_list = []
        each_step = []
        last_100_reward = []
        explore_min = 20000
        episode = 2500
        for i in range(episode):
            reset = 1
            time.sleep(0.2)
            start = time.time()
            reset = 0
            total_reward = 0
            step = 0
            difference_distance_terminal = 0
            distance_terminal, posture, distance, warning, block, overturn, reach, velocity, destination_angle = deal_main_data(get_tcp_state)
            state = np.append(distance, posture)  # 21
            state = np.append(state, destination_angle)
            state = np.append(state, velocity)
            state = np.append(state, distance_terminal)

            for t in range(500):
                action = agent.select_action(state)  # action 2维连续数值
                select_explore = [action + max(explore_min - i * 400 * (explore_min / max_action), max_action - i * 400), action + np.random.normal(0, max(1000, args.exploration_noise - i * 400), size=2),
                                  action - max(explore_min - i * 400 * (explore_min / max_action), max_action - i * 400)]
                if len(agent.memory.storage) <= (args.capacity - 1) / 20:  # 探索
                    # if t < 4000:
                    j = random.random()
                    if j < 0.6:
                        action = select_explore[0]
                        # print(0)
                    elif 0.6 <= j < 0.9:
                        action = select_explore[1]
                        # print(1)
                    else:
                        action = select_explore[2]
                        # print(2)
                action = (action + np.random.normal(0, 2000, size=2)).clip(-35000, 35000)  # 截断（-x,x）之间
                reward, restart = Reward(difference_distance_terminal, distance_terminal, destination_angle, posture, warning, block, overturn, reach, velocity)
                next_state, next_distance_terminal, next_destination_angle, posture, warning, block, overturn, reach, next_velocity = Next_state(action)
                agent.memory.push((state, next_state, action, reward, np.float(restart)))  # x是state，y是next_state，u是action，r是reward，d是done
                difference_distance_terminal = distance_terminal - next_distance_terminal  # 此时刻与下一时刻距离终点的差值
                # next
                destination_angle = next_destination_angle
                velocity = next_velocity
                distance_terminal = next_distance_terminal
                state = next_state
                if len(last_100_reward) > 100:
                    last_100_reward.pop(0)
                step += 1
                total_reward += reward

                if restart:
                    break

            left, right = 0, 0
            # 平均奖励
            mean_reward = total_reward / (step + 1)
            return_list.append(total_reward)
            mean_return_list.append(mean_reward)  # 保存到list
            last_100_reward.append(total_reward)
            # 保存每个episode的步数
            each_step.append(step)
            total_step += step + 1
            print("Total T:{} Episode: \t{} Total Reward: \t{}".format(total_step, i, total_reward))
            print("Total T:{} Episode: \t{} Mean Reward: \t{}".format(total_step, i, mean_reward))
            if len(agent.memory.storage) >= (args.capacity - 1) / 20:
                agent.update(100)
            # else:
            #     print(len(agent.memory.storage), args.capacity)

            if i % args.log_interval == 0:
                agent.save()
            last100_mean_reward = last_100_mean_reward(last_100_reward)
            if len(last_100_reward) == 100:
                print("Total T:{} Episode: \t{} Last 100 Reward: \t{}".format(total_step, i, last100_mean_reward))
            end = time.time()
            print('预计训练结束还有{}小时'.format(((end - start) * (episode - i)) / 3600))
        print("结束")
        plt.title("LOSS")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.plot(range(episode), return_list)
        return_list = pd.DataFrame(data=return_list)
        return_list.to_csv('./total_return_list1.csv', encoding='utf-8')
        mean_return_list = pd.DataFrame(data=mean_return_list)
        mean_return_list.to_csv('./mean_return_list1.csv', encoding='utf-8')
        each_step = pd.DataFrame(data=each_step)
        each_step.to_csv('./each_step1.csv', encoding='utf-8')
        plt.show()

    if args.mode == 'test':
        agent.load()
        for i in range(args.iteration):
            reset = 1
            time.sleep(0.1)
            total_reward = 0
            step = 0
            difference_distance_terminal = 0
            distance_terminal, posture, distance, warning, block, overturn, reach, velocity, destination_angle = deal_main_data(get_tcp_state)
            state = np.append(distance, posture)  # 22
            state = np.append(state, destination_angle)
            state = np.append(state, velocity)
            state = np.append(state, distance_terminal)
            for t in count():
                action = agent.select_action(state)
                reward, restart = Reward(difference_distance_terminal, distance_terminal, destination_angle, posture, warning, block, overturn, reach, velocity)
                next_state, next_distance_terminal, next_destination_angle, posture, warning, block, overturn, reach, next_velocity = Next_state(action)

                difference_distance_terminal = distance_terminal - next_distance_terminal  # 此时刻与下一时刻距离终点的差值
                step += 1
                total_reward += reward

                reset = restart
                if reset:
                    left, right = 0, 0
                    time.sleep(0.1)
                    print("Total reward \t{}, the episode is \t{:0.2f}, the step is \t{}".format(total_reward, i, t))
                    break
                # next
                destination_angle = next_destination_angle
                velocity = next_velocity
                distance_terminal = next_distance_terminal
                state = next_state
    else:
        raise NameError("mode wrong!!!")


if __name__ == '__main__':
    main()
