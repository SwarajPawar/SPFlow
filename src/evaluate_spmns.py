

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
from spn.data.metaData import get_partial_order, get_utilityNode, get_decNode, get_feature_names, get_feature_labels
from spn.data.domain_stats import get_original_stats, get_optimal_meu, get_random_policy_reward
from spn.structure.StatisticalTypes import MetaType
from spn.algorithms.SPMNDataUtil import align_data
from spn.algorithms.SPMN import SPMN
from spn.algorithms.ASPMN import Anytime_SPMN
from spn.io.Graphics import plot_spn
import matplotlib.pyplot as plt
from os import path as pth
import sys, os
import pickle


datasets = ['Export_Textiles', 'Powerplant_Airpollution', 'HIV_Screening', 'Computer_Diagnostician', 'Test_Strep', 'LungCancer_Staging']
datasets = ['Test_Strep', 'LungCancer_Staging']
path = "new_results_depth"

model_count = 11





for dataset in datasets:
	
	print(f"\n\n\n{dataset}\n\n\n")
	plot_path = f'{path}/{dataset}'

	

	#Get all the parameters
	partial_order = get_partial_order(dataset)
	utility_node = get_utilityNode(dataset)
	decision_nodes = get_decNode(dataset)
	feature_names = get_feature_names(dataset)
	feature_labels = get_feature_labels(dataset)
	meta_types = [MetaType.DISCRETE]*(len(feature_names)-1)+[MetaType.UTILITY]

	#Get Baseline stats
	original_stats = get_original_stats(dataset)
	optimal_meu = get_optimal_meu(dataset)
	random_policy_reward = get_random_policy_reward(dataset)

	all_avg_rewards = list()
	all_reward_dev = list()

	#Initialize anytime Learning
	aspmn = Anytime_SPMN(dataset, path, partial_order , decision_nodes, utility_node, feature_names, feature_labels, meta_types, cluster_by_curr_information_set=True, util_to_bin = False)
	
	#Start evaluation
	for model in range(model_count):

		#Get the model from the file
		file = open(f"{plot_path}/models/spmn_{model}.pkle","rb")
		spmn = pickle.load(file)
		file.close()

		#Evaluate the reward for the SPMNs
		avg_rewards, reward_dev = aspmn.evaluate_rewards_parallel(spmn = spmn, batch_size = 15000, batches = 15, interval_size = 5000)
		all_avg_rewards.append(avg_rewards)
		all_reward_dev.append(reward_dev)

		#Save the results to a file
		f = open(f"{plot_path}/reward_stats.txt", "w")
		f.write(f"\n\tAverage Reward : {all_avg_rewards}")
		f.write(f"\n\tReward Deviation: {all_reward_dev}")
		f.close()

		#Plot the reward
		rand_reward = np.array([random_policy_reward["reward"]]*len(all_avg_rewards))
		dev = np.array([random_policy_reward["dev"]]*len(all_avg_rewards))
		plt.fill_between(np.arange(len(all_avg_rewards)), rand_reward-dev, rand_reward+dev, alpha=0.1, color="lightgrey")
		plt.plot(rand_reward, linestyle="dashed", color ="grey", label="Random Policy")

		original_reward = np.array([original_stats["reward"]]*len(all_avg_rewards))
		dev = np.array([original_stats["dev"]]*len(all_avg_rewards))
		plt.fill_between(np.arange(len(all_avg_rewards)), original_reward-dev, original_reward+dev, alpha=0.3, color="red")
		plt.plot([optimal_meu]*len(all_avg_rewards), linewidth=3, color ="lime", label="Optimal MEU")
		plt.plot(original_reward, linestyle="dashed", color ="red", label="LearnSPMN")

		plt.errorbar(np.arange(len(all_avg_rewards)), all_avg_rewards, yerr=all_reward_dev, marker="o", label="Anytime")
		plt.title(f"{dataset} Average Rewards")
		plt.legend()
		plt.savefig(f"{plot_path}/rewards.png", dpi=100)
		plt.close()
		

		
