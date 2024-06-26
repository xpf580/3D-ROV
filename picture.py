# -*- coding: utf-8 -*-
# @Time    : 2022/4/15 8:44
# @Author  : 耀
import pandas as pd
import numpy as np
import csv
from matplotlib import pyplot as plt

c = '1'


# print('./total_return_list'+c+'.csv')
def readcsv(file):
    file = open(file, 'r', encoding="utf-8")  # 读取以utf-8
    context = file.read()  # 读取成str
    list_result = context.split("\n")  # 以回车符\n分割成单独的行
    # 每一行的各个元素是以【,】分割的，因此可以
    length = len(list_result)
    for i in range(length):
        list_result[i] = list_result[i].split(",")
    list_result.pop(0)
    list_result.pop()
    result = []
    for i in range(len(list_result)):
        result.append(float(list_result[i][1]))
    return result


reward = readcsv('./data/total_return_list' + c + '.csv')
# print(reward)
# reward = read('./total_return_list.csv')
plt.title("Total_reward")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.plot(range(len(reward)), reward, color="red")
plt.grid(True)
plt.show()

each_step = readcsv('./data/each_step' + c + '.csv')
plt.title("Each_step")
plt.xlabel("Episode")
plt.ylabel("Step")
plt.plot(range(len(each_step)), each_step, color="red")
plt.grid(True)
plt.show()

mean_reward = readcsv('./data/mean_return_list' + c + '.csv')
plt.title("Mean_reward")
plt.xlabel("Episode")
plt.ylabel("Mean_reward")
plt.plot(range(len(mean_reward)), mean_reward, color="red")
# plt.axis([0,800,-3,7])
plt.grid(True)
plt.show()
