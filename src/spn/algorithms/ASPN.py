

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
from spn.io.Graphics import plot_spn
from spn.algorithms.splitting.Conditioning import (
	get_split_rows_naive_mle_conditioning,
	get_split_rows_random_conditioning,
)

from spn.algorithms.splitting.Clustering import get_split_rows_XMeans
from spn.algorithms.splitting.RDC import get_split_cols_single_RDC_py, get_split_cols_distributed_RDC_py
from spn.algorithms.Inference import log_likelihood
import time
import pickle
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
import multiprocessing


min_instances_slice=200
threshold=0.3
ohe=False
leaves = create_histogram_leaf
rand_gen=None
cpus=-1


class AnytimeSPN:

	def __init__(self, dataset, output_path, ds_context, k=None):

		self.dataset = dataset          #Dataset Name
		self.spn = None
		self.ds_context = ds_context
				
		#Create directory for output as output_path/dataset
		if k is None:
			self.plot_path = f"{output_path}/{dataset}"
			if not pth.exists(self.plot_path):
				try:
					os.makedirs(self.plot_path)
				except OSError:
					print ("Creation of the directory %s failed" % self.plot_path)
					sys.exit()
		else:
			self.plot_path = f"{output_path}/{dataset}/{k}"
			if not pth.exists(self.plot_path):
				try:
					os.makedirs(self.plot_path)
				except OSError:
					print ("Creation of the directory %s failed" % self.plot_path)
					sys.exit()

	def get_loglikelihood(self, instance):
		test_data = np.array(instance).reshape(-1, self.max_var)
		return log_likelihood(self.spn, test_data)[0][0]

	# Learn Anytime SPN
	def learn_aspn(self, train, test=None, get_stats = False, save_models=True, batches=10):
			
	
		#Initialize max_var to total number of variables in the dataset
		self.max_var = train.shape[1]
	
		#Lists to store log-likelihoods and #nodes
		avg_ll = list()
		nodes = list()
		runtime = list()
		
		#Initialize cluster (k) limit to 2
		k_limit = 2
		
		#frame containing last 3 log-likelihoods for convergence 
		past3 = list()
	
		#Initialize number of variables for splitting to sqrt(n)
		n = int(self.max_var**0.5)  
		#Step size for increment in #variables for splitting
		step = (self.max_var - (self.max_var**0.5))/20

		#Initialize stats:
		stats = {"runtime": None,
				"ll" : None,
				"nodes" : None}

		#Create directory to save models
		if save_models:
			if not pth.exists(f'{self.plot_path}/models'):
				try:
					os.makedirs(f'{self.plot_path}/models')
				except OSError:
					print ("Creation of the directory models failed")
					sys.exit()

		i = 0
		while True:
			
			print("\n\n\n\n\n")
			print(f"Iteration {i+1}:\n")

			#Split columns using anytime RDC testing on n variables and distributing remaining variables over the clusters
			split_cols = get_split_cols_distributed_RDC_py(rand_gen=rand_gen, ohe=ohe, n_jobs=cpus, n=round(n))
			#Split rows using XMeans clustering given the cluster limit
			split_rows = get_split_rows_XMeans(limit=k_limit, returnk=False)
			
			#Initialize next operation
			nextop = get_next_operation(min_instances_slice)
			
			#Learn the SPN structure
			print("Start Learning...")
			start = time.time()
			spn = learn_structure(train, self.ds_context, split_rows, split_cols, leaves, nextop)
			end = time.time()
			print(f"SPN {i+1} learned!\n\n")
			self.spn = spn

			runtime.append(end-start)

			if save_models:
				#Save models
				file = open(f"{self.plot_path}/models/spn_{i+1}.pkle",'wb')
				pickle.dump(spn, file)
				file.close()
			
			if get_stats:
				#Get the total #nodes
				nodes.append(get_structure_stats_dict(spn)["nodes"])

						

				#Evaluate the log-likelihood over the test data
				pool = multiprocessing.Pool()
				batch_size = int(len(test)/batches)
				batch = list()
				total_ll = 0
				for j in range(batches):
					test_slice = test[j*batch_size:(j+1)*batch_size]
					lls = pool.map(self.get_loglikelihood, test_slice)
					total_ll += sum(lls)
					printProgressBar(j+1, batches, prefix = f'Evaluation Progress {i+1}:', suffix = 'Complete', length = 50)
				
				avg_ll.append(total_ll/len(test))

				#Print the stats
				
				print(f"\n\nX-Means Limit: {k_limit}, \tVariables for splitting: {round(n)}")
				print("#Nodes: ",nodes[i])
				print("Log-likelihood: ",avg_ll[i])
				print("Run Time: ",runtime[i])
				print("\n\n\n")
				
				
				#Save the stats to a file
				f = open(f"{self.plot_path}/stats.txt", "w")
				f.write(f'{self.dataset}:')
				f.write(f"\n\t#Nodes: {nodes}")
				f.write(f"\n\tLog-likelihood: {avg_ll}")
				f.write(f"\n\tRun Time: {runtime}")
				f.close()

				stats['runtime'] = runtime
				stats['ll'] = avg_ll
				stats['nodes'] = nodes

				yield (self.spn, stats)
			
			#Save the log-likelihood past 3 iterations
			past3 = avg_ll[-min(len(avg_ll),3):]
					
			#Convergence Criteria
			#If it includes all variables for splitting
			#And the std for past3 is less than 1e-3
			if n>=self.max_var and (round(np.std(past3), 3) <= 0.001 or i>=50):
				break
			
			#Increase the number of clusters and variables for splitting
			i+=1
			n = min(n+step, self.max_var)
			k_limit += 1

		

		

			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
