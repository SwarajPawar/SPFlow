

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
dataset = 'SkillTeaching'
path = "final_results"

plot_path = f"{path}/{dataset}"


runtime = [225.45611214637756, 409.30220341682434, 2561.281684398651, 3642.331939935684, 4664.270489454269, 5637.146071910858, 6784.246602296829, 7800.384288549423, 8438.307054758072, 9201.956100940704, 9266.72284913063]
avg_ll = [-24.4339512381427, -5.969381767210244, -1.9439431667940255, -1.9439431667940255, -1.9439431667940255, -1.9439431667940255, -0.8981039994673795, -0.8981039994673795, -0.8981039994673795, -0.8981039994673795, -0.8981039994673795]
ll_dev = [0.5690318459563833, 0.08113381492825335, 0.030688176346793265, 0.030688176346793265, 0.030688176346793265, 0.030688176346793265, 0.01686138579108599, 0.01686138579108599, 0.01686138579108599, 0.01686138579108599, 0.01686138579108599]
meus = [-47.1293284254487, -7.898217471735515, -7.607713884467893, -7.607713884467893, -7.607713884467893, -7.607713884467893, -7.180539, -7.180539, -7.180539, -7.180539, -7.180539]
nodes = [942, 1693, 22161, 22161, 22161, 22161, 42320, 42320, 42320, 42320, 42320]
avg_rewards = [-8.879470800541522, -7.9052637823558545, -7.615391472184976, -7.615364310784777, -7.615364310784777, -7.615374187657583, -7.180538999996855, -7.180538999996855, -7.180538999996855, -7.180538999996855, -7.180538999996855]
reward_dev = [0.019032849140173006, 0.019984646324725464, 0.011742554008000049, 0.012830753463376178, 0.013191795402603417, 0.011334614192267348, 0.0, 0.0, 0.0, 0.0, 0.0]



original_stats = get_original_stats(dataset)
optimal_meu = get_optimal_meu(dataset)
random_policy_reward = get_random_policy_reward(dataset)




plt.rc('font', size=11)          # controls default text sizes
plt.rc('axes', titlesize=14)     # fontsize of the axes title
plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=11)    # fontsize of the tick labels
plt.rc('ytick', labelsize=11)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=14)  # fontsize of the figure title


plt.close()



plt.close()
plt.plot(range(1,len(runtime)+1), [original_stats["runtime"]]*len(runtime), linestyle="dashed", linewidth=2, color ="darkred", label="LearnSPMN")
plt.plot(range(1,len(runtime)+1), runtime, marker="o", label="Anytime", color ="black", linewidth=3)
plt.title(f"{dataset} Run Time (in seconds)")
plt.xlabel("Iteration")
plt.ylabel("Run Time")
plt.legend()
plt.savefig(f"{plot_path}/runtime.png", dpi=250, bbox_inches='tight', pad_inches = 0.3)
plt.close()

plt.close()
plt.plot(range(1,len(avg_ll)+1), [original_stats["ll"]]*len(avg_ll), linestyle="dashed", linewidth=2, color ="darkred", label="LearnSPMN")
plt.errorbar(range(1,len(avg_ll)+1), avg_ll, yerr=ll_dev, marker="o", label="Anytime", color ="black", linewidth=3)
plt.title(f"{dataset} Log Likelihood")
plt.xlabel("Iteration")
plt.ylabel("Log Likelihood")
plt.legend()
plt.savefig(f"{plot_path}/ll.png", dpi=250, bbox_inches='tight', pad_inches = 0.3)
plt.close()

plt.plot(range(1,len(meus)+1), meus, marker="o", label="Anytime", color ="black", linewidth=3)
plt.plot(range(1,len(meus)+1), [optimal_meu]*len(meus), linewidth=3, color ="lime", label="Optimal MEU")
plt.plot(range(1,len(meus)+1), [original_stats["meu"]]*len(meus), linestyle="dashed", linewidth=2, color ="darkred", label="LearnSPMN")
plt.title(f"{dataset} MEU")
plt.xlabel("Iteration")
plt.ylabel("MEU")
plt.legend()
plt.savefig(f"{plot_path}/meu.png", dpi=250, bbox_inches='tight', pad_inches = 0.3)
plt.close()

plt.plot(range(1,len(nodes)+1), nodes, marker="o", label="Anytime", color ="black", linewidth=3)
plt.plot(range(1,len(nodes)+1), [original_stats["nodes"]]*len(nodes), linestyle="dashed", linewidth=2, color ="darkred", label="LearnSPMN")
plt.title(f"{dataset} Nodes")
plt.xlabel("Iteration")
plt.ylabel("# Nodes")
plt.legend()
plt.savefig(f"{plot_path}/nodes.png", dpi=250, bbox_inches='tight', pad_inches = 0.3)
plt.close()


rand_reward = np.array([random_policy_reward["reward"]]*len(avg_rewards))
dev = np.array([random_policy_reward["dev"]]*len(avg_rewards))
plt.fill_between(range(1,len(avg_rewards)+1),  rand_reward-dev, rand_reward+dev, alpha=0.1, color="lightgrey")
plt.plot(range(1,len(avg_rewards)+1), rand_reward, linestyle="dashed", linewidth=3, color ="grey", label="Random Policy")

original_reward = np.array([original_stats["reward"]]*len(avg_rewards))
dev = np.array([original_stats["dev"]]*len(avg_rewards))
plt.fill_between(range(1,len(avg_rewards)+1), original_reward-dev, original_reward+dev, alpha=0.3, color="red")
plt.plot(range(1,len(avg_rewards)+1), [optimal_meu]*len(avg_rewards), linewidth=5, color ="lime", label="Optimal MEU")
plt.plot(range(1,len(avg_rewards)+1), original_reward, linestyle="dashed", linewidth=2, color ="darkred", label="LearnSPMN")

plt.errorbar(range(1,len(avg_rewards)+1), avg_rewards, yerr=reward_dev, marker="o", label="Anytime", color ="black", linewidth=3,)
plt.title(f"{dataset} Average Rewards")
plt.xlabel("Iteration")
plt.ylabel("Average Rewards")
plt.legend()
plt.savefig(f"{plot_path}/rewards.png", dpi=250, bbox_inches='tight', pad_inches = 0.3)
plt.close()
