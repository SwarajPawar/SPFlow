

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
dataset = 'CrossingTraffic'
path = "test"
path = "new_results_depth1"

plot_path = f"{path}/{dataset}"

runtime = [367.90279507637024, 806.3270936012268, 3164.336494445801, 3874.411342382431, 4633.373266220093, 5392.799260139465, 5523.524729251862, 11957.648134231567, 12560.33174777031, 12428.710459947586, 6803.5443415641785]
avg_ll = [-12.62200935452218, -7.172130296998749, -4.485157790960345, -4.485157790960345, -4.485157790960345, -4.485157790960345, -3.0328539659488234, -3.0328539659488234, -3.0328539659488234, -3.0328539659488234, -3.0328539659488234]
ll_dev= [0.46294919289399167, 0.06970015640902255, 0.04424240994764077, 0.04424240994764077, 0.04424240994764077, 0.04424240994764077, 0.03784934657653855, 0.03784934657653855, 0.03784934657653855, 0.03784934657653855, 0.03784934657653855]
meus = [-4.884400462053216, -4.880978572368808, -4.690673447301574, -4.690673447301574, -4.690673447301574, -4.690673447301574, -4.049331963001028, -4.049331963001028, -4.049331963001028, -4.049331963001028, -4.049331963001028]
nodes = [2990, 13500, 140129, 140129, 140129, 140129, 198840, 198840, 198840, 198840, 198840]
#edges = [792, 4163, 11285, 11285, 11279, 11279, 15274, 15274, 15274, 15274, 15274]
#layers = [16, 24, 22, 22, 22, 22, 16, 16, 16, 16, 16]
#avg_rewards = [-4.946, -5.0, -5.0, -5.0, -5.0, -5.0, -4.0416, -4.0408, -4.0408, -4.0408, -4.0412]
#reward_dev = [0.02788548009269327, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03444183502660688, 0.03669550381177512, 0.03742940020892686, 0.035611234182487995, 0.03665187580465679]


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
plt.xlabel("Iteration")
plt.ylabel("Run Time")
plt.legend()
plt.savefig(f"{plot_path}/runtime.png", dpi=100)
plt.close()

plt.close()
plt.plot(range(1,len(avg_ll)+1), [original_stats["ll"]]*len(avg_ll), linestyle="dotted", color ="red", label="LearnSPMN")
plt.errorbar(range(1,len(avg_ll)+1), avg_ll, yerr=ll_dev, marker="o", label="Anytime")
plt.title(f"{dataset} Log Likelihood")
plt.xlabel("Iteration")
plt.ylabel("Log Likelihood")
plt.legend()
plt.savefig(f"{plot_path}/ll.png", dpi=100)
plt.close()

plt.plot(range(1,len(meus)+1), meus, marker="o", label="Anytime")
plt.plot(range(1,len(meus)+1), [optimal_meu]*len(meus), linewidth=3, color ="lime", label="Optimal MEU")
plt.plot(range(1,len(meus)+1), [original_stats["meu"]]*len(meus), linestyle="dotted", color ="red", label="LearnSPMN")
plt.title(f"{dataset} MEU")
plt.xlabel("Iteration")
plt.ylabel("MEU")
plt.legend()
plt.savefig(f"{plot_path}/meu.png", dpi=100)
plt.close()

plt.plot(range(1,len(nodes)+1), nodes, marker="o", label="Anytime")
plt.plot(range(1,len(nodes)+1), [original_stats["nodes"]]*len(nodes), linestyle="dotted", color ="red", label="LearnSPMN")
plt.title(f"{dataset} Nodes")
plt.xlabel("Iteration")
plt.ylabel("# Nodes")
plt.legend()
plt.savefig(f"{plot_path}/nodes.png", dpi=100)
plt.close()
'''
plt.plot(range(1,len(edges)+1), edges, marker="o", label="Anytime")
plt.plot(range(1,len(edges)+1), [original_stats["edges"]]*len(edges), linestyle="dotted", color ="red", label="LearnSPMN")
plt.title(f"{dataset} Edges")
plt.xlabel("Iteration")
plt.ylabel("# Edges")
plt.legend()
plt.savefig(f"{plot_path}/edges.png", dpi=100)
plt.close()

plt.plot(range(1,len(layers)+1), layers, marker="o", label="Anytime")
plt.plot(range(1,len(layers)+1), [original_stats["layers"]]*len(layers), linestyle="dotted", color ="red", label="LearnSPMN")
plt.title(f"{dataset} Layers")
plt.xlabel("Iteration")
plt.ylabel("# Layers")
plt.legend()
plt.savefig(f"{plot_path}/layers.png", dpi=100)
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
'''