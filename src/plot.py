

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
dataset = 'Test_Strep'
path = "test"
path = "latest_rewards"

plot_path = f"{path}/{dataset}"

avg_rewards = [54.913958064009, 54.91025275200896, 54.91251739200899, 54.913229328008974, 54.91814772000897, 54.938849796008974, 54.93647870400901, 54.938862948008975, 54.92918382000897]
reward_dev = [0.011204484209889187, 0.01460571111198627, 0.01406649716560237, 0.014319393880995351, 0.012567516348352976, 0.00862493156540575, 0.009851411824126095, 0.01110178741069467, 0.011171635839162697]

original_stats = {
	'Export_Textiles': {"ll" : -1.0890750655173789, "meu" : 1722313.8158882717, 'nodes' : 38, 'reward':1721301.8260000004, 'dev':3861.061525772288},
	'Test_Strep': {"ll" : -0.9130071749277912, "meu" : 54.9416526618876, 'nodes' : 130, 'reward':54.936719928008955, 'dev':0.011715846521357575},
	'LungCancer_Staging': {"ll" : -1.1489156814245234, "meu" : 3.138664586296027, 'nodes' : 312, 'reward':3.1429205999999272, 'dev':0.01190315582691798},
	'HIV_Screening': {"ll" : -0.6276399171508842, "meu" : 42.582734183407034, 'nodes' : 112, 'reward':42.559788119992646, 'dev':0.06067708771159484},
	'Computer_Diagnostician': {"ll" : -0.9011245432112749, "meu" : -208.351, 'nodes' : 56, 'reward':-210.13350000000002, 'dev':0.4155875359054929},
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

rand_reward = np.array([random_reward[dataset]["reward"]]*len(avg_rewards))
dev = np.array([random_reward[dataset]["dev"]]*len(avg_rewards))
plt.fill_between(np.arange(len(avg_rewards)), rand_reward-dev, rand_reward+dev, alpha=0.1, color="lightgrey")
plt.plot(rand_reward, linestyle="dashed", color ="grey", label="Random Policy")

original_reward = np.array([original_stats[dataset]["reward"]]*len(avg_rewards))
dev = np.array([original_stats[dataset]["dev"]]*len(avg_rewards))
plt.fill_between(np.arange(len(avg_rewards)), original_reward-dev, original_reward+dev, alpha=0.3, color="red")
plt.plot([optimal_meu[dataset]]*len(avg_rewards), linewidth=3, color ="lime", label="Optimal MEU")
plt.plot(original_reward, linestyle="dashed", color ="red", label="LearnSPMN")
plt.errorbar(np.arange(len(avg_rewards)), avg_rewards, yerr=reward_dev, marker="o", label="Anytime")
plt.title(f"{dataset} Average Rewards")
plt.legend()
plt.savefig(f"{plot_path}/final_rewards.png", dpi=100)
plt.close()