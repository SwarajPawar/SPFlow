

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
import time

datasets = ['Computer_Diagnostician',  'Test_Strep', 'LungCancer_Staging']
#datasets = ['Export_Textiles','HIV_Screening, 'Powerplant_Airpollution', ]
datasets = ['Navigation' ]
path = "original_new"

def get_loglikelihood(instance):
	test_data = np.array(instance).reshape(-1, len(feature_names))
	return log_likelihood(spmn, test_data)[0][0]

def get_reward(ids):

	#policy = ""
	state = env.reset()
	while(True):
		output = best_next_decision(spmn, state)
		action = output[0][0]
		#policy += f"{action}  "
		state, reward, done = env.step(action)
		if done:
			return reward
			#return policy



for dataset in datasets:
	
	print(f"\n\n\n{dataset}\n\n\n")
	plot_path = f"{path}/{dataset}"
	if not pth.exists(plot_path):
		try:
			os.makedirs(plot_path)
		except OSError:
			print ("Creation of the directory %s failed" % plot_path)
			sys.exit()


	partial_order = get_partial_order(dataset)
	utility_node = get_utilityNode(dataset)
	decision_nodes = get_decNode(dataset)
	feature_names = get_feature_names(dataset)
	feature_labels = get_feature_labels(dataset)
	meta_types = [MetaType.DISCRETE]*(len(feature_names)-1)+[MetaType.UTILITY]

			
	df = pd.read_csv(f"spn/data/{dataset}/{dataset}.tsv", sep='\t')

	df, column_titles = align_data(df, partial_order)  # aligns data in partial order sequence
	
	data = df.values
	train, test = data, random.sample(list(data), 50000)

	print("Start Learning...")
	spmn = SPMN(partial_order , decision_nodes, utility_node, feature_names, meta_types, cluster_by_curr_information_set = True, util_to_bin = False)
	start = time.time()
	spmn = spmn.learn_spmn(train)
	end = time.time()
	print("Done")


	runtime = end - start

	file = open(f"{path}/spmn.pkle",'wb')
	pickle.dump(spmn, file)
	file.close()
	
	
	nodes = get_structure_stats_dict(spmn)["nodes"]
	
	if nodes <= 500:
		plot_spn(spmn, f'{path}/{dataset}/spmn.pdf', feature_labels=feature_labels)


	pool = multiprocessing.Pool()

	
	
	batch_size = int(test.shape[0] / 10)
	total_ll = 0
	test = list(test)
	for b in range(10):
		test_slice = test[b*batch_size:(b+1)*batch_size]
		lls = pool.map(get_loglikelihood, test_slice)
		total_ll += sum(lls)
		printProgressBar(b+1, 10, prefix = f'Log Likelihood Evaluation :', suffix = 'Complete', length = 50)
	
	'''
	for j, instance in enumerate(test):
		test_data = np.array(instance).reshape(-1, len(feature_names))
		total_ll += log_likelihood(spmn, test_data)[0][0]
		printProgressBar(j+1, len(test), prefix = f'Log Likelihood Evaluation :', suffix = 'Complete', length = 50)
	
	'''
	ll = (total_ll/len(test))
	
	
	test_data = [[np.nan]*len(feature_names)]
	m = meu(spmn, test_data)
	meus = (m[0])
	
	'''
	env = get_env(dataset)
	total_reward = 0
	batch_count = 25
	batch_size = 20000 
	batch = list()

	pool = multiprocessing.Pool()
	policy_set = list()

	for z in range(batch_count):
		
		ids = [None for x in range(batch_size)]
		rewards = pool.map(get_reward, ids)
		

		#policies = pool.map(get_reward, ids)
		#policy_set += policies
		#print(Counter(policy_set))
		
		batch.append(sum(rewards)/batch_size)
		#print(batch[-1])
		printProgressBar(z+1, batch_count, prefix = f'Average Reward Evaluation :', suffix = 'Complete', length = 50)

	
	avg_rewards = np.mean(batch)
	reward_dev = np.std(batch)
	'''

	print(f"\n\tRun Time: {runtime}")
	print(f"\n\tLog Likelihood : {ll}")
	print(f"\n\tMEU : {meus}")
	print(f"\n\tNodes : {nodes}")
	#print(f"\n\tAverage rewards : {avg_rewards}")
	#print(f"\n\tDeviation : {reward_dev}")
	
	
	f = open(f"{path}/{dataset}/stats.txt", "w")
	f.write(f"\n{dataset}")
	f.write(f"\n\tRun Time : {runtime}")
	f.write(f"\n\tLog Likelihood : {ll}")
	f.write(f"\n\tMEU : {meus}")
	f.write(f"\n\tNodes : {nodes}")
	#f.write(f"\n\tAverage rewards : {avg_rewards}")
	#f.write(f"\n\tDeviation : {reward_dev}")
	f.close()
	
	

