

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


datasets = ['Export_Textiles']
path = "test"







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
	df = pd.read_csv(f"spn/data/{dataset}/{dataset}_new.tsv", sep='\t')
	df, column_titles = align_data(df, partial_order)
	data = df.values

	test_size = int(data.shape[0]*0.1)
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
	for i, output in enumerate(aspmn.learn_aspmn(train, get_stats=False)):

		spmn, stats = output

		f = open(f"{plot_path}/stats.txt", "a")
		f.write(f"\n\n\n\n")
		f.close()
		#Plot the SPMN
		#plot_spn(spmn, f'{plot_path}/spmn{i}.pdf', feature_labels=feature_labels)

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



		