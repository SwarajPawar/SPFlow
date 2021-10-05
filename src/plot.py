

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
dataset = 'GameOfLife'
path = "new_results_depth"

plot_path = f"{path}/{dataset}"

'''
runtime = [225.45611214637756, 409.30220341682434, 2561.281684398651, 3642.331939935684, 4664.270489454269, 5637.146071910858, 6784.246602296829, 7800.384288549423, 8438.307054758072, 9201.956100940704, 9266.72284913063]
avg_ll = [-24.4339512381427, -5.969381767210244, -1.9439431667940255, -1.9439431667940255, -1.9439431667940255, -1.9439431667940255, -0.8981039994673795, -0.8981039994673795, -0.8981039994673795, -0.8981039994673795, -0.8981039994673795]
ll_dev = [0.5690318459563833, 0.08113381492825335, 0.030688176346793265, 0.030688176346793265, 0.030688176346793265, 0.030688176346793265, 0.01686138579108599, 0.01686138579108599, 0.01686138579108599, 0.01686138579108599, 0.01686138579108599]
meus = [-47.1293284254487, -7.898217471735515, -7.607713884467893, -7.607713884467893, -7.607713884467893, -7.607713884467893, -7.180539, -7.180539, -7.180539, -7.180539, -7.180539]
nodes = [942, 1693, 22161, 22161, 22161, 22161, 42320, 42320, 42320, 42320, 42320]
edges = [792, 4163, 11285, 11285, 11279, 11279, 15274, 15274, 15274, 15274, 15274]
layers = [12, 16, 22, 22, 22, 22, 16, 16, 16, 16, 16]
'''
avg_rewards = [9.481524, 9.664282, 9.088116, 9.814084000000001, 10.006813999999999, 10.275244, 10.425716, 10.808250000000003, 10.858739999999997, 10.808398000000002, 10.808460000000002, 10.808449999999999, 10.808534, 10.808398000000002, 10.808476, 10.808398, 10.808356000000002, 10.80827]
reward_dev = [0.028922140722982607, 0.03161312980392801, 0.029373735615342032, 0.03335275616796883, 0.016855453242200065, 0.02568124732173254, 0.03489619096692368, 0.027985696346526607, 0.0314709008450663, 0.0281872718793429, 0.028684619572167957, 0.028427546499830016, 0.027994244837109036, 0.028776457321915114, 0.02840035253302317, 0.028688260595581574, 0.028019148166923215, 0.028887384789904293]



original_stats = get_original_stats(dataset)
optimal_meu = get_optimal_meu(dataset)
random_policy_reward = get_random_policy_reward(dataset)


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
'''

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
