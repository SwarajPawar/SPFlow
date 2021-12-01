

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
dataset = 'Navigation'
path = "output"

plot_path = f"{path}/{dataset}"




original_stats = get_original_stats(dataset)
optimal_meu = get_optimal_meu(dataset)
random_policy_reward = get_random_policy_reward(dataset)


k_ll = [[-3.9932947638998693, -2.040437607001177, -1.38227977348897, -1.38227977348897, -1.38227977348897, -1.38227977348897, -0.2362233894707108, -0.2362233894707108, -0.2362233894707108, -0.2362233894707108, -0.2362233894707108],
				[-3.9932947638998693, -2.040437607001177, -1.38227977348897, -1.38227977348897, -1.38227977348897, -1.38227977348897, -0.2362233894707108, -0.2362233894707108, -0.2362233894707108, -0.2362233894707108, -0.2362233894707108],
				[-4.14497743616636, -2.080353698600632, -1.4292936317061433, -1.4292936317061433, -1.4292936317061433, -1.4292936317061433, -0.24283866624677644, -0.24283866624677644, -0.24283866624677644, -0.24283866624677644, -0.24283866624677644]]





plt.rc('font', size=11)          # controls default text sizes
plt.rc('axes', titlesize=14)     # fontsize of the axes title
plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=11)    # fontsize of the tick labels
plt.rc('ytick', labelsize=11)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=14)  # fontsize of the figure title


plt.close()



plt.close()
colors = ["red", "blue", "green"]

maxlen = max([len(k_ll[i]) for i in range(len(k_ll))])
total_ll = np.zeros(min([len(k_ll[i]) for i in range(len(k_ll))]))
originalll = [original_stats["ll"]] * maxlen
plt.plot(range(1, maxlen+1), originalll, linestyle="dotted", color ="purple", linewidth=3, label="LearnSPN")
for i in range(len(k_ll)):
	plt.plot(range(1,len(k_ll[i])+1), k_ll[i], marker=f"{i+1}", color =colors[i], label=(i+1))
	total_ll += np.array(k_ll[i][:len(total_ll)])
avg_ll = total_ll/len(k_ll)
plt.plot(range(1,len(avg_ll)+1), avg_ll, marker="o", color ="black", linewidth=1, label="Mean")
plt.title(f"{dataset} Log Likelihood")
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Log Likelihood")
plt.savefig(f"{path}/{dataset}/ll_{dataset}.png", dpi=150)
plt.close()