

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
import pickle

datasets = ['Export_Textiles']
path = "output"

#Get Log-likelihood
def get_loglikelihood(instance):
	test_data = np.array(instance).reshape(-1, len(feature_names))
	return log_likelihood(spmn, test_data)[0][0]

#Get Reward by simulating policy
def get_reward(ids):

	state = env.reset()
	while(True):
		output = best_next_decision(spmn, state)
		action = output[0][0]
		state, reward, done = env.step(action)
		if done:
			return reward



	
#Create Output Directory
print(f"\n\n\n{dataset}\n\n\n")
plot_path = f"{path}/{dataset}"
if not pth.exists(plot_path):
	try:
		os.makedirs(plot_path)
	except OSError:
		print ("Creation of the directory %s failed" % plot_path)
		sys.exit()


#Get required Parameters
partial_order = get_partial_order(dataset)
utility_node = get_utilityNode(dataset)
decision_nodes = get_decNode(dataset)
feature_names = get_feature_names(dataset)
feature_labels = get_feature_labels(dataset)
meta_types = [MetaType.DISCRETE]*(len(feature_names)-1)+[MetaType.UTILITY]

		
#Read dataset
df = pd.read_csv(f"spn/data/{dataset}/{dataset}.tsv", sep='\t')
#Align data
df, column_titles = align_data(df, partial_order)  # aligns data in partial order sequence


data = df.values
train, test = data, data

#Start Learning SPMN
print("Start Learning...")
spmn = SPMN(partial_order , decision_nodes, utility_node, feature_names, meta_types, cluster_by_curr_information_set = True, util_to_bin = False)
start = time.time()
spmn = spmn.learn_spmn(train)
end = time.time()
print("Done")

#Save the model
file = open(f"{path}/{dataset}/spmn_original.pkle",'wb')
pickle.dump(spmn, file)
file.close()


#Plot SPMN
plot_spn(spmn, f'{path}/{dataset}/spmn.pdf', feature_labels=feature_labels)


#Compute Log-likelihood
pool = multiprocessing.Pool()

batch_size = int(test.shape[0] / 10)
total_ll = 0
test = list(test)
for b in range(10):
	test_slice = test[b*batch_size:(b+1)*batch_size]
	lls = pool.map(get_loglikelihood, test_slice)
	total_ll += sum(lls)
	printProgressBar(b+1, 10, prefix = f'Log Likelihood Evaluation :', suffix = 'Complete', length = 50)
ll = (total_ll/len(test))

#Get MEU
test_data = [[np.nan]*len(feature_names)]
m = meu(spmn, test_data)
meus = (m[0])


#Get environment for reward
env = get_env(dataset)
total_reward = 0
batch_count = 25
batch_size = 20000 
batch = list()

pool = multiprocessing.Pool()
policy_set = list()

#Compute Average Reward
for z in range(batch_count):
	ids = [None for x in range(batch_size)]
	rewards = pool.map(get_reward, ids)
	batch.append(sum(rewards)/batch_size)
	printProgressBar(z+1, batch_count, prefix = f'Average Reward Evaluation :', suffix = 'Complete', length = 50)
avg_rewards = np.mean(batch)
reward_dev = np.std(batch)

#Print Results
print(f"\n\tLog Likelihood : {ll}")
print(f"\n\tMEU : {meus}")
print(f"\n\tAverage rewards : {avg_rewards}")
print(f"\n\tDeviation : {reward_dev}")
	
	
	
	

