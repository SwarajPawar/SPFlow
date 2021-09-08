

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
dataset = 'GameOfLife'
path = "test"
path = "new_results_depth1"

plot_path = f"{path}/{dataset}"

runtime = [152.5618634223938, 752.8968393802643, 3778.9327857494354, 5492.052267074585, 9348.144783735275, 12224.432264566422, 13196.357997894287, 14114.30926990509, 14051.375816106796, 15170.068479061127, 15279.90201663971, 15503.499443531036, 15483.576924562454, 15515.465003490448, 15589.440560340881]
avg_ll = [-10.853872823727759, -10.966744863853283, -8.403255377986866, -8.267961821772527, -6.943724413973098, -5.983020631214568, -5.8859795001001265, -5.146443214290348, -5.112322679251555, -4.626295210955778, -4.6522117494988855, -4.663144855047961, -4.635588395486958, -4.630584907877058, -4.631948083465982]
ll_dev= [0.18449233769708268, 0.2588791078811392, 0.2915910361391348, 0.2552993504069091, 0.2025085427481468, 0.1250056661790957, 0.14439783769741152, 0.2581449405276246, 0.266010639161589, 0.2583059104608148, 0.25988660855080575, 0.2492470895382542, 0.24874832178756212, 0.24496720762962396, 0.25242020788056896]
meus = [77.420539264374, 103.95818818414463, 86.99235833541321, 84.05180702707186, 77.91300167805147, 80.17449749361604, 83.64804323267263, 85.95626580090102, 85.956265800901, 85.30956279842715, 85.20554813803666, 85.18909897894765, 85.18909897894767, 85.18909897894768, 85.18909897894765]
nodes = [1378, 14504, 187264, 254698, 437152, 540562, 563413, 615900, 617588, 685351, 686571, 684710, 684457, 684110, 684121]
edges = [1291, 14314, 177620, 239929, 407713, 502078, 522758, 570655, 572255, 634129, 635263, 633501, 633261, 632926, 632936]
layers = [18, 21, 34, 37, 41, 39, 38, 40, 43, 39, 41, 39, 39, 39, 39]
avg_rewards = [9.6004, 10.145599999999998, 9.25, 9.802, 10.286, 10.034, 9.38, 10.66, 10.664, 10.663999999999998, 10.664, 10.664, 10.663999999999998, 10.668, 10.663999999999998]
reward_dev = [0.15122645271248045, 0.09533435896884153, 0.2830547650190688, 0.23181026724457227, 0.11774548823628031, 0.24385241438214186, 0.28955137713366197, 0.35349681752457124, 0.35617972991173985, 0.3819738210924932, 0.37754999668918043, 0.3590877330124213, 0.3819738210924932, 0.3876286883087994, 0.3819738210924932]



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
