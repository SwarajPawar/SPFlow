

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
import matplotlib.pyplot as plt
from os import path as pth
import sys, os


datasets = ['Elevators']
path = "new_results_depth1"







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

	#Get test and train data
	df = pd.read_csv(f"spn/data/{dataset}/{dataset}.tsv", sep='\t')
	df, column_titles = align_data(df, partial_order)
	data = df.values

	test_size = int(data.shape[0]*0.02)
	train, test = data, np.array(random.sample(list(data), test_size))

	#Initialize anytime Learning
	aspmn = Anytime_SPMN(dataset, path, partial_order , decision_nodes, utility_node, feature_names, feature_labels, meta_types, cluster_by_curr_information_set=True, util_to_bin = False)
	

	f = open(f"{plot_path}/stats.txt", "w")
	f.write(f"\n{dataset}")
	f.close()

	#Initialize lists for storing statistics over iterations
	all_avg_ll = list()
	all_ll_dev = list()
	all_meus = list()
	all_nodes = list()
	all_avg_rewards = list()
	all_reward_dev = list()


	#Start anytime learning
	for i, output in enumerate(aspmn.learn_aspmn(train, test, get_stats=True, evaluate_parallel=True)):

		spmn, stats = output

		f = open(f"{plot_path}/stats.txt", "a")
		f.write(f"\n\n\n\n")
		f.close()
		
		#Plot the SPMN
		#plot_spn(spmn, f'{plot_path}/spmn{i}.pdf', feature_labels=feature_labels)

		#Get stats
		runtime = stats["runtime"]
		avg_ll = stats["ll"]
		ll_dev = stats["ll_dev"]
		meus = stats["meu"]
		nodes = stats["nodes"]

		# plot the statistics

		plt.close()
		plt.plot(runtime, marker="o", label="Anytime")
		plt.title(f"{dataset} Run Time (in seconds)")
		plt.legend()
		plt.savefig(f"{plot_path}/runtime.png", dpi=100)
		plt.close()

		plt.close()
		plt.errorbar(np.arange(len(avg_ll)), avg_ll, yerr=ll_dev, marker="o", label="Anytime")
		plt.title(f"{dataset} Log Likelihood")
		plt.legend()
		plt.savefig(f"{plot_path}/ll.png", dpi=100)
		plt.close()
		
		plt.plot(meus, marker="o", label="Anytime")
		plt.title(f"{dataset} MEU")
		plt.legend()
		plt.savefig(f"{plot_path}/meu.png", dpi=100)
		plt.close()

		plt.plot(nodes, marker="o", label="Anytime")
		plt.title(f"{dataset} Nodes")
		plt.legend()
		plt.savefig(f"{plot_path}/nodes.png", dpi=100)
		plt.close()

		'''
		#Get stats
		all_runtime = stats["runtime"]
		f = open(f"{plot_path}/stats.txt", "a")
		f.write(f"\n\tRuntime : {all_runtime}")
		f.close()

		#Get stats
		all_nodes.append(aspmn.evaluate_nodes(spmn))
		f = open(f"{plot_path}/stats.txt", "a")
		f.write(f"\n\tNodes : {all_nodes}")
		f.close()

		all_meus.append(aspmn.evaluate_meu(spmn))
		f = open(f"{plot_path}/stats.txt", "a")
		f.write(f"\n\tMEU : {all_meus}")
		f.close()

		
		avg_ll, ll_dev = aspmn.evaluate_loglikelihood_sequential(test, spmn)
		all_avg_ll.append(avg_ll)
		all_ll_dev.append(ll_dev)
		f = open(f"{plot_path}/stats.txt", "a")
		f.write(f"\n\tLog Likelihood : {all_avg_ll}")
		f.write(f"\n\tLog Likelihood Deviation: {all_ll_dev}")
		f.close()
		'''


		
