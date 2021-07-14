

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
dataset = 'Elevators'
path = "test"
path = "latest_rewards"

plot_path = f"{path}/{dataset}"

reward = [-6, -6, -3.6100000000000003, -6.015, -5.26]
dev = [0, 0, 3.1304791965448358, 2.6918441633943075, 1.8615584868598674]


plt.close()
'''
rand_reward = np.array([random_reward[dataset]["reward"]]*len(avg_rewards))
dev = np.array([random_reward[dataset]["dev"]]*len(avg_rewards))
plt.fill_between(np.arange(len(avg_rewards)), rand_reward-dev, rand_reward+dev, alpha=0.1, color="lightgrey")
plt.plot(rand_reward, linestyle="dashed", color ="grey", label="Random Policy")

original_reward = np.array([original_stats[dataset]["reward"]]*len(avg_rewards))
dev = np.array([original_stats[dataset]["dev"]]*len(avg_rewards))
plt.fill_between(np.arange(len(avg_rewards)), original_reward-dev, original_reward+dev, alpha=0.3, color="red")
plt.plot([optimal_meu[dataset]]*len(avg_rewards), linewidth=3, color ="lime", label="Optimal MEU")
plt.plot(original_reward, linestyle="dashed", color ="red", label="LearnSPMN")

plt.errorbar(np.arange(len(avg_rewards)), avg_rewards, yerr=reward_dev, marker="o", label="Anytime1")

plt.errorbar(np.arange(len(avg_rewards1)), avg_rewards1, yerr=reward_dev1, marker="o", label="Anytime1")
plt.errorbar(np.arange(len(avg_rewards2)), avg_rewards2, yerr=reward_dev2, marker="o", label="Anytime2")
plt.errorbar(np.arange(len(avg_rewards3)), avg_rewards3, yerr=reward_dev3, marker="o", label="Anytime3")

mean_rewards = [0]*len(avg_rewards1)
for i in range(len(mean_rewards)):
	mean_rewards[i] = (avg_rewards1[i] + avg_rewards2[i] + avg_rewards3[i]) / 3
plt.plot(mean_rewards, marker="o", color ="black", label="Anytime mean")


plt.title(f"{dataset} Average Rewards")
plt.legend()
plt.savefig(f"{plot_path}/rewards_final.png", dpi=100)
plt.close()
'''


plt.close()
plt.errorbar(np.arange(len(reward)), reward, yerr=dev, marker="o", label="Anytime")
plt.title(f"{dataset} Average Rewards")
plt.legend()
plt.savefig(f"{plot_path}/rewards.png", dpi=100)
plt.close()

