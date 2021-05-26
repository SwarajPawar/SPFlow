

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
datasets = ['Computer_Diagnostician']
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


avg_rewards = [244.74949999999998, 244.7045, 244.88199999999998, 244.8545, 242.9505, 242.98850000000002, 245.10999999999999, 245.05550000000002, 244.74650000000003, 244.8955]
reward_dev =  [0.5562169091280859, 0.4264296542221232, 0.28283564131842037, 0.38524959441899526, 0.6986503059471153, 0.5492770248244502, 0.5577633906953691, 0.6370967351980423, 0.3354180227715824, 0.6508742197998008]




original_reward = np.array([original_stats[self.dataset]["reward"]]*len(avg_rewards))
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
if original_reward[0] > 0:
	plt.axis(ymin=minl, ymax=maxl)
else:
	plt.axis(ymax=minl, ymin=maxl)
plt.title(f"{dataset} Average Rewards")
plt.legend()
plt.savefig(f"{plot_path}/rewards_scaled3.png", dpi=100)
plt.close()