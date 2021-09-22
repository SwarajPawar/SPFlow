

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


#datasets = ['Export_Textiles', 'Test_Strep', 'HIV_Screening', 'Computer_Diagnostician', 'Powerplant_Airpollution', 'LungCancer_Staging']
dataset = 'CrossingTraffic'
path = "test"
path = "new_results_depth1"

plot_path = f"{path}/{dataset}"

runtime = [273.6596667766571, 2998.2903044223785, 2402.158529281616, 9973.869557619095, 12207.940668344498, 14397.18154501915, 14421.774231672287, 15663.243483304977]
avg_ll = [-18.207073210453355, -6.916125229487657, -8.276427809509395, -4.288879929509539, -4.288879929509539, -4.288879929509539, -3.0386144623173807, -3.0386144623173807]
ll_dev= [0.28872423214851506, 0.05175015174657449, 0.036820542664803176, 0.02337240315391185, 0.02337240315391185, 0.02337240315391185, 0.015015841271669514, 0.015015841271669514]
meus = [-5.0, -4.616392555275525, -4.425137515746642, -4.352325246892821, -4.352325246892821, -4.352325246892821, -4.0, -4.0]
nodes = [1706, 67203, 20584, 193392, 193392, 193392, 273206, 273206]
edges = [1611, 63616, 19254, 181131, 181131, 181131, 254527, 254527]
layers = [18, 30, 26, 30, 30, 30, 20, 20]
avg_rewards = [-4.98107, -4.9559180000000005, -4.824955999999999, -4.825236, -4.825672, -4.824771999999999, -4.0, -4.0, -4.0]
reward_dev = [0.0007544534445543898, 0.0014944483932207168, 0.0029884885812062945, 0.0028405464263059786, 0.0026177883795294975, 0.002926023923347266, 0.0, 0.0, 0.0]



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
plt.xlabel("Iteration")
plt.ylabel("Average Rewards")
plt.legend()
plt.savefig(f"{plot_path}/rewards.png", dpi=100)
plt.close()
