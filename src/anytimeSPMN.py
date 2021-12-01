

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
from spn.algorithms.AnytimeSPMN import Anytime_SPMN
from spn.io.Graphics import plot_spn
import matplotlib.pyplot as plt
from os import path as pth
import sys, os


datasets = ['Export_Textiles', 'Powerplant_Airpollution', 'HIV_Screening', 'Computer_Diagnostician', 'Test_Strep', 'LungCancer_Staging']
path = "output"







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

	#Get test and train data
	df = pd.read_csv(f"spn/data/{dataset}/{dataset}.tsv", sep='\t')
	df, column_titles = align_data(df, partial_order)
	data = df.values

	train, test = data, data

	#Initialize anytime Learning
	aspmn = Anytime_SPMN(dataset, path, partial_order , decision_nodes, utility_node, feature_names, feature_labels, meta_types, cluster_by_curr_information_set=True, util_to_bin = False)
	
	#Start anytime learning
	for i, output in enumerate(aspmn.anytime_learn_spmn(train, test, get_stats=True, evaluate_parallel=True)):

		spmn, stats = output

		#Plot the SPMN
		plot_spn(spmn, f'{plot_path}/spmn{i}.pdf', feature_labels=feature_labels)

		#Get stats
		runtime = stats["runtime"]
		avg_ll = stats["ll"]
		ll_dev = stats["ll_dev"]
		meus = stats["meu"]
		nodes = stats["nodes"]
		parameters = stats["parameters"]
		layers = stats["layers"]
		avg_rewards = stats["reward"]
		reward_dev = stats["reward_dev"]

		# plot the statistics

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

		plt.plot(range(1,len(parameters)+1), parameters, marker="o", label="Anytime")
		plt.plot(range(1,len(parameters)+1), [original_stats["parameters"]]*len(parameters), linestyle="dotted", color ="red", label="LearnSPMN")
		plt.title(f"{dataset} Parameters")
		plt.xlabel("Iteration")
		plt.ylabel("# Parameters")
		plt.legend()
		plt.savefig(f"{plot_path}/parameters.png", dpi=100)
		plt.close()

		plt.plot(range(1,len(layers)+1), layers, marker="o", label="Anytime")
		plt.plot(range(1,len(layers)+1), [original_stats["layers"]]*len(layers), linestyle="dotted", color ="red", label="LearnSPMN")
		plt.title(f"{dataset} Layers")
		plt.xlabel("Iteration")
		plt.ylabel("# Layers")
		plt.legend()
		plt.savefig(f"{plot_path}/layers.png", dpi=100)
		plt.close()


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



		
