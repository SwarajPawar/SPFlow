

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
from spn.algorithms.MEU import meu
from spn.algorithms.Inference import log_likelihood
from spn.algorithms.Statistics import get_structure_stats_dict
from spn.io.Graphics import plot_spn
from spn.data.simulator import get_env
from spn.algorithms.MEU import best_next_decision
from spn.io.ProgressBar import printProgressBar

import multiprocessing
import matplotlib.pyplot as plt
from os import path as pth
import sys, os
from collections import Counter
import random

datasets = ['HIV_Screening',  'Test_Strep', 'LungCancer_Staging']
datasets = {'Export_Textiles':3,'Computer_Diagnostician':2, 'Powerplant_Airpollution':2, 'HIV_Screening':2,  'Test_Strep':2, 'LungCancer_Staging':2}
#datasets = ['Test_Strep' ]
path = "original_new"



def get_reward(ids):

	policy = ""
	state = env.reset()
	while(True):
		
		action = random.randint(0, datasets[dataset]-1)
		policy += f"{action}  "
		state, reward, done = env.step(action)
		'''
		if action==1:
			print(state)
			#
		'''
		if done:
			return reward
			return policy



for dataset in datasets:
	
	print(f"\n\n\n{dataset}\n\n\n")
	plot_path = f"{path}/{dataset}"
	if not pth.exists(plot_path):
		try:
			os.makedirs(plot_path)
		except OSError:
			print ("Creation of the directory %s failed" % plot_path)
			sys.exit()


	
	
	env = get_env(dataset)
	total_reward = 0
	#trials = 200000
	batch_count = 25
	batch_size = 20000 #int(trials / batch_count)
	batch = list()

	pool = multiprocessing.Pool()
	policy_set = list()

	for z in range(batch_count):
		
		ids = [None for x in range(batch_size)]
		rewards = pool.map(get_reward, ids)
		
		batch.append(sum(rewards)/batch_size)
		#print(batch[-1])
		printProgressBar(z+1, batch_count, prefix = f'Average Reward Evaluation :', suffix = 'Complete', length = 50)

	
	avg_rewards = np.mean(batch)
	reward_dev = np.std(batch)
	
	
	print(f"\n\tAverage rewards : {avg_rewards}")
	print(f"\n\tDeviation : {reward_dev}")
	
	
	f = open(f"{path}/{dataset}/random.txt", "w")
	f.write(f"\n{dataset}")
	
	f.write(f"\n\tAverage rewards : {avg_rewards}")
	f.write(f"\n\tDeviation : {reward_dev}")
	f.close()
	
	

