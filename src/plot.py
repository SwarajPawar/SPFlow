

import numpy as np

import logging
logger = logging.getLogger(__name__)


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from os import path as pth
import sys, os
import random
from spn.data.domain_stats import get_original_stats, get_optimal_meu, get_random_policy_reward

import matplotlib.pyplot as plt
from os import path as pth
import sys, os


datasets = ['Export_Textiles', 'Test_Strep', 'HIV_Screening', 'Computer_Diagnostician', 'Powerplant_Airpollution', 'LungCancer_Staging']
dataset = 'Navigation'
path = "test"
path = "new_results_depth"

plot_path = f"{path}/{dataset}"

runtime = [145.29267168045044, 436.09778928756714, 891.3127875328064, 782.6671736240387, 1089.476714372635, 1329.4888541698456, 1294.7908515930176, 1529.7081279754639, 1747.1985025405884, 1898.2388129234314, 2076.967685699463]
avg_ll = [-4.0880219823450314, -2.049680238096561, -1.3969299582391717, -1.3969299582391717, -1.3969299582391717, -1.3969299582391717, -0.25178044936277855, -0.25178044936277855, -0.25178044936277855, -0.25178044936277855, -0.25178044936277855]
ll_dev= [0.09862152329505344, 0.07712501063510499, 0.04257641737421496, 0.04257641737421496, 0.04257641737421496, 0.04257641737421496, 0.031783895937702894, 0.031783895937702894, 0.031783895937702894, 0.031783895937702894, 0.031783895937702894]
meus = [-4.776024118453267, -4.765703262088994, -4.396269763478818, -4.396269763478818, -4.396269763478818, -4.396269763478818, -3.120315581854044, -3.120315581854044, -3.120315581854044, -3.120315581854044, -3.120315581854044]
nodes = [893, 4621, 12512, 12512, 12507, 12507, 17338, 17338, 17338, 17338, 17338]
avg_rewards = [-4.887199999999999, -5.0, -5.0, -5.0, -5.0, -5.0, -3.104, -3.1033999999999997, -3.1043999999999996, -3.1042, -3.1033999999999997]
reward_dev = [0.04472314836860214, 0.0, 0.0, 0.0, 0.0, 0.0, 0.040477154050155256, 0.040968768592673156, 0.041063852717444796, 0.04119660180160496, 0.04040841496520253]

original_stats = get_original_stats(dataset)
optimal_meu = get_optimal_meu(dataset)
random_policy_reward = get_random_policy_reward(dataset)


plt.close()


'''
plt.close()
plt.errorbar(np.arange(len(reward)), reward, yerr=dev, marker="o", label="Anytime")
plt.title(f"{dataset} Average Rewards")
plt.legend()
plt.savefig(f"{plot_path}/rewards.png", dpi=100)
plt.close()
'''

plt.close()
plt.plot(range(1,len(runtime)+1), [original_stats["runtime"]]*len(runtime), linestyle="dotted", color ="red", label="LearnSPMN")
plt.plot(range(1,len(runtime)+1), runtime, marker="o", label="Anytime")
plt.title(f"{dataset} Run Time (in seconds)")
plt.legend()
plt.savefig(f"{plot_path}/runtime.png", dpi=100)
plt.close()

plt.close()
plt.plot(range(1,len(avg_ll)+1), [original_stats["ll"]]*len(avg_ll), linestyle="dotted", color ="red", label="LearnSPMN")
plt.errorbar(range(1,len(avg_ll)+1), avg_ll, yerr=ll_dev, marker="o", label="Anytime")
plt.title(f"{dataset} Log Likelihood")
plt.legend()
plt.savefig(f"{plot_path}/ll.png", dpi=100)
plt.close()

plt.plot(range(1,len(meus)+1), meus, marker="o", label="Anytime")
plt.plot(range(1,len(meus)+1), [optimal_meu]*len(meus), linewidth=3, color ="lime", label="Optimal MEU")
plt.plot(range(1,len(meus)+1), [original_stats["meu"]]*len(meus), linestyle="dotted", color ="red", label="LearnSPMN")
plt.title(f"{dataset} MEU")
plt.legend()
plt.savefig(f"{plot_path}/meu.png", dpi=100)
plt.close()

plt.plot(range(1,len(nodes)+1), nodes, marker="o", label="Anytime")
plt.plot(range(1,len(nodes)+1), [original_stats["nodes"]]*len(nodes), linestyle="dotted", color ="red", label="LearnSPMN")
plt.title(f"{dataset} Nodes")
plt.legend()
plt.savefig(f"{plot_path}/nodes.png", dpi=100)
plt.close()

rand_reward = np.array([random_policy_reward["reward"]]*len(avg_rewards))
dev = np.array([random_policy_reward["dev"]]*len(avg_rewards))
plt.fill_between(range(1,len(avg_rewards)+1),  rand_reward-dev, rand_reward+dev, alpha=0.1, color="lightgrey")
plt.plot(range(1,len(avg_rewards)+1), rand_reward, linestyle="dashed", color ="grey", label="Random Policy")

original_reward = np.array([original_stats["reward"]]*len(avg_rewards))
dev = np.array([original_stats["dev"]]*len(avg_rewards))
plt.fill_between(range(1,len(avg_rewards)+1), original_reward-dev, original_reward+dev, alpha=0.3, color="red")
plt.plot(range(1,len(avg_rewards)+1), [optimal_meu]*len(avg_rewards), linewidth=3, color ="lime", label="Optimal MEU")
plt.plot(range(1,len(avg_rewards)+1), original_reward, linestyle="dashed", color ="red", label="LearnSPMN")

plt.errorbar(range(1,len(avg_rewards)+1), avg_rewards, yerr=reward_dev, marker="o", label="Anytime")
plt.title(f"{dataset} Average Rewards")
plt.legend()
plt.savefig(f"{plot_path}/rewards.png", dpi=100)
plt.close()
