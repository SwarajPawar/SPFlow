

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
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import pickle


datasets = ['Navigation', 'SkillTeaching', 'CrossingTraffic', 'GameOfLife']
path = "output"

model_count = {'Navigation':11,
				'SkillTeaching': 11,
				'CrossingTraffic': 11,
				'GameOfLife': 18}


kfold = KFold(n_splits=3, shuffle=True)





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

	k_ll = list()
	k_dev = list()

	k = 1
	for trainidx, testidx in kfold.split(data):
		#train, test = train_test_split(data, test_size=0.3, shuffle=True)

		print(f"\n\n\n{k}\n\n")

		if dataset in ['SkillTeaching'] and k==1:
			k+=1
			continue

		
		train, test = data[trainidx], data[testidx]
		test = np.array(random.sample(list(test), 500))

		plot_path = f"{path}/{dataset}/{k}"
		if not pth.exists(plot_path):
			try:
				os.makedirs(plot_path)
			except OSError:
				print ("Creation of the directory %s failed" % plot_path)
				sys.exit()

		#Initialize anytime Learning
		aspmn = Anytime_SPMN(dataset, plot_path, partial_order , decision_nodes, utility_node, feature_names, feature_labels, meta_types, cluster_by_curr_information_set=True, util_to_bin = False)
		
		'''
		#Start anytime learning
		for i, output in enumerate(aspmn.anytime_learn_spmn(train, test, get_stats=True, evaluate_parallel=True, log_likelihood_batches=5, save_models=False)):

			spmn, stats = output

			#Get stats
			#Get stats
			runtime = stats["runtime"]
			avg_ll = stats["ll"]
			ll_dev = stats["ll_dev"]
		'''
			#Start evaluation
		avg_ll = list()
		ll_dev = list()
		prev_size = 0

		for model in range(model_count[dataset]):

			#Get the model from the file
			file = open(f"AnytimeSPMN_results/{dataset}/models/spmn_{model+1}.pkle","rb")
			spmn = pickle.load(file)
			file.close()

			struct_stats = aspmn.evaluate_structure_stats(spmn = spmn)
			nodes = struct_stats["nodes"]

			if nodes == prev_size:
				ll = avg_ll[-1]
				dev = ll_dev[-1]
			else:
				ll, dev = aspmn.evaluate_loglikelihood_parallel(test, spmn=spmn, batches=5)

			avg_ll.append(ll)
			ll_dev.append(dev)
			prev_size = nodes

			f = open(f"{plot_path}/stats.txt", "w")
			f.write(f"\n\tLog-Likelihood : {avg_ll}")
			f.write(f"\n\tLikelihood Deviation: {ll_dev}")
			f.close()


			plt.close()
			plt.plot(range(1,len(avg_ll)+1), [original_stats["ll"]]*len(avg_ll), linestyle="dotted", color ="red", label="LearnSPMN")
			plt.errorbar(range(1,len(avg_ll)+1), avg_ll, yerr=ll_dev, marker="o", label="Anytime")
			plt.title(f"{dataset} Log Likelihood")
			plt.xlabel("Iteration")
			plt.ylabel("Log Likelihood")
			plt.legend()
			plt.savefig(f"{plot_path}/ll.png", dpi=100)
			plt.close()

		k_ll.append(avg_ll)
		k_dev.append(ll_dev)

		f = open(f"{path}/{dataset}/stats.txt", "w")
		f.write(f"\n\tLog-Likelihood : {avg_ll}")
		f.write(f"\n\tLikelihood Deviation: {ll_dev}")
		f.close()

		k+=1

		