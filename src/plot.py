

import numpy as np

import logging
logger = logging.getLogger(__name__)


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from os import path as pth
import sys, os
import random

from sklearn.model_selection import train_test_split
from spn.data.metaData import *
from spn.structure.StatisticalTypes import MetaType
from spn.algorithms.SPMNDataUtil import align_data
from spn.algorithms.SPMN import SPMN
from spn.algorithms.ASPMN import Anytime_SPMN
import matplotlib.pyplot as plt
from os import path as pth
import sys, os


datasets = ['Export_Textiles', 'Test_Strep', 'HIV_Screening', 'Computer_Diagnostician', 'Powerplant_Airpollution', 'LungCancer_Staging']
dataset = 'Test_Strep'
path = "test"
path = "improve"

plot_path = f"{path}/{dataset}"

original_stats = {
	'Export_Textiles': {"ll" : -1.0890750655173789, "meu" : 1722313.8158882717, 'nodes' : 38, 'reward':1716130.8399999999, 'dev':8877.944736840887},
	'Test_Strep': {"ll" : -0.9130071749277912, "meu" : 54.9416526618876, 'nodes' : 130, 'reward':54.93578280000071, 'dev':0.018246756840598732},
	'LungCancer_Staging': {"ll" : -1.1489156814245234, "meu" : 3.138664586296027, 'nodes' : 312, 'reward':3.1265179999999946, 'dev':0.024158974233189766},
	'HIV_Screening': {"ll" : -0.6276399171508842, "meu" : 42.582734183407034, 'nodes' : 112, 'reward':42.64759879999822, 'dev':0.13053757307440556},
	'Computer_Diagnostician': {"ll" : -0.8920749045689644, "meu" : 244.85700000000003, 'nodes' : 47, 'reward':245.04599999999996, 'dev':0.40763218714915067},
	'Powerplant_Airpollution': {"ll" : -1.0796486063753, "meu" : -2756263.244346315, 'nodes' : 46, 'reward':-2750100.0, 'dev':25448.182646310914}
}


avg_rewards = [54.90729528000175, 54.908305200001735, 54.9034046400018, 54.91022004000183, 54.91520016000182, 54.91820376000178, 54.922305600001756, 54.94782636000175, 54.936296640001764, 54.933508440001845, 54.93837408000174]
reward_dev =  [0.02338342514646736, 0.034278461999997345, 0.03196405985992598, 0.024855123251643362, 0.02770146877621585, 0.022704493693135052, 0.018163959038125257, 0.01936005652767528, 0.01935357306400966, 0.022176988608176934, 0.017021098611604075]


original_reward = np.array([original_stats[dataset]["reward"]]*len(avg_rewards))
dev = np.array([original_stats[dataset]["dev"]]*len(avg_rewards))
lspmn_reward = str(abs(int(original_reward[0])))
order = len(lspmn_reward)
r_dev = np.array(reward_dev)
if order > 1:
	 minl= (round(min(avg_rewards-r_dev)/(10**(order-2)) * 2)/2 - 0.5) * (10**(order-2))
	 maxl= (round(max(avg_rewards+r_dev)/(10**(order-2)) * 2)/2 + 0.5) * (10**(order-2))
else:
	minl= round(min(avg_rewards-r_dev)*2)/2 - 0.5
	maxl= round(max(avg_rewards+r_dev)*2)/2 + 0.5
plt.plot(original_reward, linestyle="dotted", color ="red", label="LearnSPMN")
plt.fill_between(np.arange(len(avg_rewards)), original_reward-dev, original_reward+dev, alpha=0.3, color="red")
plt.errorbar(np.arange(len(avg_rewards)), avg_rewards, yerr=reward_dev, marker="o", label="Anytime")
plt.axis(ymin=minl, ymax=maxl)
plt.title(f"{dataset} Average Rewards")
plt.legend()
plt.savefig(f"{plot_path}/rewards_scaled3.png", dpi=100)
plt.close()