

import numpy as np

import logging
logger = logging.getLogger(__name__)


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from os import path as pth
import sys, os
import random


import matplotlib.pyplot as plt
from os import path as pth
import sys, os


datasets = ['Export_Textiles', 'Test_Strep', 'HIV_Screening', 'Computer_Diagnostician', 'Powerplant_Airpollution', 'LungCancer_Staging']
dataset = 'Computer_Diagnostician'
path = "test"
path = "latest_rewards"

plot_path = f"{path}/{dataset}"

avg_rewards = [-210.15184999999997, -210.271, -210.15205000000003, -209.96450000000002, -209.95780000000002, -210.06165000000001, -210.14565]
reward_dev = [0.43855517326785537, 0.3888174378805566, 0.3919931249397103, 0.3539634515031189, 0.39943245361387547, 0.4315781678908229, 0.45885606675732094]

avg_rewards2 = [1720097.1639999999, 1723201.61, 1722467.999, 1720807.31, 1722290.493]
reward_dev2 = [4289.37011657354, 5005.749572816257, 3861.9006886083857, 4595.004550291557, 4149.031242913946]

avg_rewards3 = [1720335.251, 1722878.24, 1723119.596, 1720900.157, 1721280.4309999999]
reward_dev3 = [3618.798937986745, 5494.864819824963, 3983.324125279559, 4087.8948808067485, 5494.616497173761]

original_stats = {
	'Export_Textiles': {"ll" : -1.0890750655173789, "meu" : 1722313.8158882717, 'nodes' : 38, 'reward':1721301.8260000004, 'dev':3861.061525772288},
	'Test_Strep': {"ll" : -0.9130071749277912, "meu" : 54.9416526618876, 'nodes' : 130, 'reward':54.91352060400901, 'dev':0.013189836549851251},
	'LungCancer_Staging': {"ll" : -1.1489156814245234, "meu" : 3.138664586296027, 'nodes' : 312, 'reward':3.108005299999918, 'dev':0.011869627022775012},
	'HIV_Screening': {"ll" : -0.6276399171508842, "meu" : 42.582734183407034, 'nodes' : 112, 'reward':42.559788119992646, 'dev':0.06067708771159484},
	'Computer_Diagnostician': {"ll" : -0.9011245432112749, "meu" : -208.351, 'nodes' : 56, 'reward':-210.15520000000004, 'dev':0.3810022440878799},
	'Powerplant_Airpollution': {"ll" : -1.0796885930912947, "meu" : -2756263.244346315, 'nodes' : 38, 'reward':-2759870.4, 'dev':6825.630813338794}
}

optimal_meu = {
	'Export_Textiles' : 1721300,
	'Computer_Diagnostician': -210.13,
	'Powerplant_Airpollution': -2760000,
	'HIV_Screening': 42.5597,
	'Test_Strep': 54.9245,
	'LungCancer_Staging': 3.12453
}

random_reward = {
	'Export_Textiles' : {'reward': 1300734.02, 'dev':7087.350616838437},
	'Computer_Diagnostician': {'reward': -226.666, 'dev':0.37205611135956335},
	'Powerplant_Airpollution': {'reward': -3032439.0, 'dev':7870.276615214995},
	'HIV_Screening': {'reward': 42.3740002199867, 'dev':0.07524234474837802},
	'Test_Strep': {'reward': 54.89614493400057, 'dev':0.012847272731391593},
	'LungCancer_Staging': {'reward': 2.672070640000026, 'dev':0.007416967451081523},
}

plt.close()
'''
rand_reward = np.array([random_reward[dataset]["reward"]]*len(avg_rewards))
dev = np.array([random_reward[dataset]["dev"]]*len(avg_rewards))
plt.fill_between(np.arange(len(avg_rewards)), rand_reward-dev, rand_reward+dev, alpha=0.1, color="lightgrey")
plt.plot(rand_reward, linestyle="dashed", color ="grey", label="Random Policy")
'''
original_reward = np.array([original_stats[dataset]["reward"]]*len(avg_rewards))
dev = np.array([original_stats[dataset]["dev"]]*len(avg_rewards))
plt.fill_between(np.arange(len(avg_rewards)), original_reward-dev, original_reward+dev, alpha=0.3, color="red")
plt.plot([optimal_meu[dataset]]*len(avg_rewards), linewidth=3, color ="lime", label="Optimal MEU")
plt.plot(original_reward, linestyle="dashed", color ="red", label="LearnSPMN")

plt.errorbar(np.arange(len(avg_rewards)), avg_rewards, yerr=reward_dev, marker="o", label="Anytime1")
'''
plt.errorbar(np.arange(len(avg_rewards1)), avg_rewards1, yerr=reward_dev1, marker="o", label="Anytime1")
plt.errorbar(np.arange(len(avg_rewards2)), avg_rewards2, yerr=reward_dev2, marker="o", label="Anytime2")
plt.errorbar(np.arange(len(avg_rewards3)), avg_rewards3, yerr=reward_dev3, marker="o", label="Anytime3")

mean_rewards = [0]*len(avg_rewards1)
for i in range(len(mean_rewards)):
	mean_rewards[i] = (avg_rewards1[i] + avg_rewards2[i] + avg_rewards3[i]) / 3
plt.plot(mean_rewards, marker="o", color ="black", label="Anytime mean")
'''

plt.title(f"{dataset} Average Rewards")
plt.legend()
plt.savefig(f"{plot_path}/rewards_final.png", dpi=100)
plt.close()