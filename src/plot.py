

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

runtime = [283.31241512298584, 783.5959687232971, 846.5336625576019, 789.5973365306854, 1084.5150356292725, 1327.8394014835358, 1308.8737914562225, 1525.1038265228271, 1769.5259737968445, 1913.9400238990784, 2042.3172421455383]
avg_ll = [-4.116424278933279, -2.113518334819445, -1.4393207685426315, -1.4393207685426315, -1.4393207685426315, -1.4393207685426315, -0.25974613589634254, -0.25974613589634254, -0.25974613589634254, -0.25974613589634254, -0.25974613589634254]
ll_dev= [0.07563352462533411, 0.06690529782580432, 0.036895127955141944, 0.036895127955141944, 0.036895127955141944, 0.036895127955141944, 0.023240346211611484, 0.023240346211611484, 0.023240346211611484, 0.023240346211611484, 0.023240346211611484]
meus = [-4.884400462053216, -4.880978572368808, -4.690673447301574, -4.690673447301574, -4.690673447301574, -4.690673447301574, -4.049331963001028, -4.049331963001028, -4.049331963001028, -4.049331963001028, -4.049331963001028]
nodes = [893, 4578, 12681, 12681, 12675, 12675, 17426, 17426, 17426, 17426, 17426]
edges = [792, 4163, 11285, 11285, 11279, 11279, 15274, 15274, 15274, 15274, 15274]
layers = [16, 24, 22, 22, 22, 22, 16, 16, 16, 16, 16]
avg_rewards = [-4.946, -5.0, -5.0, -5.0, -5.0, -5.0, -4.0416, -4.0408, -4.0408, -4.0408, -4.0412]
reward_dev = [0.02788548009269327, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03444183502660688, 0.03669550381177512, 0.03742940020892686, 0.035611234182487995, 0.03665187580465679]


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
plt.legend()
plt.savefig(f"{plot_path}/rewards.png", dpi=100)
plt.close()
