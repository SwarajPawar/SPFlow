

import numpy as np

from spn.algorithms.StructureLearning import get_next_operation, learn_structure
from spn.algorithms.CnetStructureLearning import get_next_operation_cnet, learn_structure_cnet
from spn.algorithms.Validity import is_valid
from spn.algorithms.Statistics import get_structure_stats_dict

from spn.structure.Base import Sum, assign_ids

from spn.structure.leaves.histogram.Histograms import create_histogram_leaf
from spn.structure.leaves.parametric.Parametric import create_parametric_leaf
from spn.structure.leaves.piecewise.PiecewiseLinear import create_piecewise_leaf
from spn.structure.leaves.cltree.CLTree import create_cltree_leaf
from spn.algorithms.splitting.Conditioning import (
	get_split_rows_naive_mle_conditioning,
	get_split_rows_random_conditioning,
)

from spn.algorithms.splitting.Clustering import get_split_rows_XMeans
from spn.algorithms.splitting.RDC import get_split_cols_single_RDC_py, get_split_cols_distributed_RDC_py

import logging

logger = logging.getLogger(__name__)


import warnings

warnings.filterwarnings('ignore')



import pandas as pd
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType
from spn.io.ProgressBar import printProgressBar
import matplotlib.pyplot as plt
from os import path as pth
import sys, os
import random


min_instances_slice=200
threshold=0.3
ohe=False
leaves = create_histogram_leaf
rand_gen=None
cpus=-1


class AnytimeSPN:

	def __init__(self, dataset, output_path, ds_context):

		self.dataset = dataset
		self.spn = None
		self.ds_context = ds_context
				
		print(f"\n\n\n{dataset}\n\n\n")
		if not pth.exists(f"{output_path}/{dataset}"):
			try:
				os.makedirs(f"{output_path}/{dataset}")
			except OSError:
				print ("Creation of the directory %s failed" % plot_path)
				sys.exit()


	def learn_aspn(self, train, test, k=None):
			
	
		self.max_iter = train.shape[1]
	

		ll = list()
		nodes = list()
		k_limit = 2 #[i for i in range(1,5)]
		past3 = list()
	
		n = int(max_iter**0.5)  
		step = (max_iter - (max_iter**0.5))/20

		i = 0
		while True:
			split_cols = get_split_cols_distributed_RDC_py(rand_gen=rand_gen, ohe=ohe, n_jobs=cpus, n=round(n))
			split_rows = get_split_rows_XMeans(limit=k_limit, returnk=False)
			nextop = get_next_operation(min_instances_slice)

			spn = learn_structure(data, ds_context, split_rows, split_cols, leaves, nextop)
			self.spn = spn
			
			nodes.append(get_structure_stats_dict(spn)["nodes"])
			from spn.io.Graphics import plot_spn

			if k is None:
				plot_spn(spn, f'{output_path}/{dataset}/spn{i}.png')
			else:
				plot_spn(spn, f'{output_path}/{dataset}/{k}/spn{i}.png')

			from spn.algorithms.Inference import log_likelihood
			total_ll = 0
			for j, instance in enumerate(test):
				test_data = np.array(instance).reshape(-1, var)
				total_ll += log_likelihood(spn, test_data)[0][0]
				printProgressBar(j+1, len(test), prefix = f'Evaluation Progress {i}:', suffix = 'Complete', length = 50)
			ll.append(total_ll/len(test))

			
			print("\n\n\n\n\n")
			print("Iteration {i}:\n")
			print(f"X-Means Limit: {k_limit}, \tVariables for splitting: {round(n)}")
			print("#Nodes: ",nodes[i])
			print("Log-likelihood: ",ll[i])
			print("\n\n\n\n\n")
			
			plt.close()
			plt.plot(ll, marker="o") 
			plt.title(f"{dataset} Log Likelihood")
			plt.savefig(f"{output_path}/{dataset}/ll.png", dpi=100)
			if k is None:
				plt.savefig(f"{output_path}/{dataset}/ll.png", dpi=100)
			else:
				plt.savefig(f"{output_path}/{dataset}/{k}/ll.png", dpi=100)
			plt.close()
			
			plt.plot(nodes, marker="o") 
			plt.title(f"{dataset} Nodes")
			if k is None:
				plt.savefig(f"{output_path}/{dataset}/nodes.png", dpi=100)
			else:
				plt.savefig(f"{output_path}/{dataset}/{k}/nodes.png", dpi=100)
			plt.close()

			if k is None:
				f = open(f"{output_path}/{dataset}/stats.txt", "w")
			else:
				f = open(f"{output_path}/{dataset}/{k}/stats.txt", "w")
			f.write(f"\n\tLog Likelihood: {ll}")
			f.write(f"\n\t\tNodes: {nodes}")
			f.close()
			
			past3 = ll[-min(len(ll),3):]
					
			if n>=max_iter and round(np.std(past3), 3) <= 0.001:
				break
			
			i+=1
			n = min(n+step, max_iter)
			k_limit += 1

		print("Final interation")
		print("Log Likelihood",ll)
		print("Nodes",nodes)

	def learn_aspn_kfold(self, data, k):
	
		from sklearn.model_selection import KFold
		
		kfold = KFold(n_splits=k, shuffle=True)
		
		k = 1
		for trainidx, testidx in kfold.split(data):
			
			train, test = data[trainidx], data[testidx]
			self.learn_aspn(train, test, k=k)
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
