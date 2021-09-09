'''

This Code is used to learn and evaluate
the SPN models for the given datasets
for the maximum possible parameters for 
the anytime splitting techniques
using the LearnSPN algorithm


'''

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
from spn.algorithms.Statistics import get_structure_stats_dict
from spn.io.ProgressBar import printProgressBar
import matplotlib.pyplot as plt
from os import path as pth
import sys, os
import time
import multiprocessing

#Initialize parameters

min_instances_slice=200
threshold=0.3
ohe=False
leaves = create_histogram_leaf
rand_gen=None
cpus=-1

datasets = ["nltcs","msnbc", "plants", "kdd", "baudio", "jester", "bnetflix"]
path = "maxlimit_new"


#Get log-likelihood for the instance
def get_loglikelihood(instance):
	test_data = np.array(instance).reshape(-1, var)
	return log_likelihood(spn, test_data)[0][0]





#Leanr SPNs for each dataset
for dataset in datasets:
	
	print(f"\n\n\n{dataset}\n\n\n")
	
	#Create output directory
	if not pth.exists(path):
		try:
			os.makedirs(path)
		except OSError:
			print ("Creation of the directory %s failed" % path)
			sys.exit()
	
	#Read training and test datasets
	#Read training and test datasets
	df = pd.read_csv(f"spn/data/binary/{dataset}.ts.data", sep=',')
	data1 = df.values
	df2 = pd.read_csv(f"spn/data/binary/{dataset}.test.data", sep=',')
	data2 = df2.values
	data = np.concatenate((data1, data2))
	var = data.shape[1]

	train, test = data, data

	ds_context = Context(meta_types=[MetaType.DISCRETE]*train.shape[1])
	ds_context.add_domains(train)

	#Set anytime splitting parameters to max values
	split_cols = get_split_cols_distributed_RDC_py(rand_gen=rand_gen, ohe=ohe, n_jobs=cpus, n=var)
	split_rows = get_split_rows_XMeans(limit=1000, returnk=False)
	nextop = get_next_operation(min_instances_slice)

	#Learn the spn
	print('\nStart Learning...')
	start = time.time()
	spn = learn_structure(train, ds_context, split_rows, split_cols, leaves, nextop)
	end = time.time()
	print('SPN Learned!\n')

	from spn.io.Graphics import plot_spn
	#Plot spn
	plot_spn(spn, f'{path}/{dataset}_spn.pdf')

	#Get nodes
	nodes = get_structure_stats_dict(spn)["nodes"]

	#Calculate log-likelihood over test data and get #nodes
	from spn.algorithms.Inference import log_likelihood

	batches = 10
	pool = multiprocessing.Pool()
	batch_size = int(len(test)/batches)
	batch = list()
	total_ll = 0
	for j in range(batches):
		test_slice = test[j*batch_size:(j+1)*batch_size]
		lls = pool.map(get_loglikelihood, test_slice)
		total_ll += sum(lls)
		printProgressBar(j+1, batches, prefix = f'Evaluation Progress:', suffix = 'Complete', length = 50)
	
	ll = total_ll/len(test)
		
	#Print and save stats
	print("\n\n\n\n\n")
	print("#Nodes: ",nodes)
	print("Log-likelihood: ",ll)
	print("Run Time: ",end-start)
	print("\n\n\n\n\n")
	f = open(f"{path}/{dataset}_stats.txt", "w")
	f.write(f"\n\n\n{dataset}\n\n")
	f.write(f"#Nodes: {nodes}")
	f.write(f"Log-likelihood: {ll}")
	f.write(f"Time: {end-start}")
	f.write("\n\n\n\n\n")
	f.close()