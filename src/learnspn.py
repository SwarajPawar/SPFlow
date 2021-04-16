

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

from spn.algorithms.splitting.Clustering import get_split_rows_KMeans
from spn.algorithms.splitting.RDC import get_split_cols_single_RDC_py, get_split_cols_RDC_py

import logging

logger = logging.getLogger(__name__)


import warnings

warnings.filterwarnings('ignore')



import pandas as pd
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType
import matplotlib.pyplot as plt
from os import path as pth
import sys, os
import time

cols="rdc"
rows="kmeans"
min_instances_slice=200
threshold=0.3
ohe=False
leaves = create_histogram_leaf
rand_gen=None
cpus=-1


datasets = ["nltcs","msnbc", "plants", "kdd", "baudio", "jester", "bnetflix"]
datasets = ["plants"]
path = "original"



for dataset in datasets:
	

	print(f"\n\n\n{dataset}\n\n\n")
	f = open("out.txt", "a")
	f.write(f"\n\n\n{dataset}\n\n\n")
	f.close()
	plot_path = f"{path}/{dataset}"
	if not pth.exists(plot_path):
		try:
			os.makedirs(plot_path)
		except OSError:
			print ("Creation of the directory %s failed" % plot_path)
			sys.exit()
			
	df = pd.read_csv(f"spn/data/binary/{dataset}.ts.data", sep=',')
	data = df.values
	print(data.shape)
	max_iter = data.shape[1]
	samples, var = data.shape
	ds_context = Context(meta_types=[MetaType.DISCRETE]*var)
	ds_context.add_domains(data)

	df2 = pd.read_csv(f"spn/data/binary/{dataset}.test.data", sep=',')
	test = df2.values
	print(test.shape)



	split_cols = get_split_cols_RDC_py(rand_gen=rand_gen, ohe=ohe, n_jobs=cpus)
	split_rows = get_split_rows_KMeans()
	nextop = get_next_operation(min_instances_slice)
	'''
	spn = learn_structure(data, ds_context, split_rows, split_cols, leaves, nextop)

	from spn.io.Graphics import plot_spn

	plot_spn(spn, f'{path}/{dataset}_spn.png')

	from spn.algorithms.Inference import log_likelihood
	total_ll = 0
	for instance in test:
		import numpy as np
		test_data = np.array(instance).reshape(-1, var)
		total_ll += log_likelihood(spn, test_data)[0][0]
	ll = total_ll/len(test)
	nodes = get_structure_stats_dict(spn)["nodes"]
		
	print("\n\n\n\n\n")
	print("#Nodes: ",nodes)
	print("Log-likelihood: ",ll)
	print("\n\n\n\n\n")
	f = open("out.txt", "a")
	f.write("\n\n\n\n\n")
	f.write(f"#Nodes: {nodes}")
	f.write(f"Log-likelihood: {ll}")
	f.write("\n\n\n\n\n")
	f.close()
	   
	'''
	start = time.time()
	spn = learn_structure(data, ds_context, split_rows, split_cols, leaves, nextop)
	end = time.time()
	print(end-start)
	'''
	f = open(f"{path}/runtimes.txt", "a")
	f.write("")
	f.write(f"\tTime: {end-start}")
	f.write("\n\n")
	f.close()
	'''



