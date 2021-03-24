

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
from sklearn.model_selection import KFold
import logging
import random
from sklearn.model_selection import train_test_split

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


min_instances_slice=200
threshold=0.3
ohe=False
leaves = create_histogram_leaf
rand_gen=None
cpus=-1


datasets = ["nltcs", "plants", "baudio", "jester", "bnetflix"]

#datasets = ["kdd"]
path = "cross_new"

#kfolds = 3
kfold = KFold(n_splits=3, shuffle=True)

for dataset in datasets:
	
	print(f"\n\n\n{dataset}\n\n\n")
	plot_path = f"{path}/{dataset}"
	if not pth.exists(plot_path):
		try:
			os.makedirs(plot_path)
		except OSError:
			print ("Creation of the directory %s failed" % plot_path)
			sys.exit()
			
	df = pd.read_csv(f"spn/data/binary/{dataset}.ts.data", sep=',')
	data1 = df.values
	print(data1.shape)
	df2 = pd.read_csv(f"spn/data/binary/{dataset}.test.data", sep=',')
	data2 = df2.values
	print(data2.shape)
	data = np.concatenate((data1, data2))





	max_iter = data.shape[1]
	rows, var = data.shape
	ds_context = Context(meta_types=[MetaType.DISCRETE]*var)
	ds_context.add_domains(data)

	lls = list()
	nodes_k = list()
	
	
	#for k in range(1,kfolds+1):
	
	k = 1
	for train, test in kfold.split(data):
		#train, test = train_test_split(data, test_size=0.3, shuffle=True)
		#test = np.array(random.sample(list(test), 1500))

		plot_path = f"{path}/{dataset}/{k}"
		if not pth.exists(plot_path):
			try:
				os.makedirs(plot_path)
			except OSError:
				print ("Creation of the directory %s failed" % plot_path)
				sys.exit()
		

		ll = list()
		nodes = list()
		k1 = 2 #[i for i in range(1,5)]
		past3 = list()
		
		n = int(max_iter**0.5)  #[i for i in range(int(max_iter**0.5),max_iter+1,2)]
		step = (max_iter - (max_iter**0.5))/20

		i = 0
		while True:
			split_cols = get_split_cols_distributed_RDC_py(rand_gen=rand_gen, ohe=ohe, n_jobs=cpus, n=round(n))
			split_rows = get_split_rows_XMeans(limit=k1, returnk=False)
			nextop = get_next_operation(min_instances_slice)

			spn = learn_structure(train, ds_context, split_rows, split_cols, leaves, nextop)

			nodes.append(get_structure_stats_dict(spn)["nodes"])
			from spn.io.Graphics import plot_spn

			plot_spn(spn, f'{path}/{dataset}/{k}/spn{i}.png')

			from spn.algorithms.Inference import log_likelihood
			total_ll = 0
			for j, instance in enumerate(test):
				import numpy as np
				test_data = np.array(instance).reshape(-1, var)
				total_ll += log_likelihood(spn, test_data)[0][0]
				printProgressBar(j+1, len(test), prefix = f'Evaluation Progress {i}:', suffix = 'Complete', length = 50)
			ll.append(total_ll/len(test))
			
			'''
			if len(ll)>3:
				past3 = ll[-3:]
				if round(np.std(past3), 2) <= 0.01:
					break

			'''
			if n==max_iter:
				break
			
			print("\n\n\n\n\n")
			print(k1,round(n))
			print(nodes[i])
			print(ll[i])
			print(ll)
			print(nodes)
			print("\n\n\n\n\n")
			
			
			plt.close()
			# plot line 
			plt.plot(ll, marker="o") 
			plt.title(f"{dataset} Log Likelihood")
			plt.savefig(f"{path}/{dataset}/{k}/ll.png", dpi=100)
			plt.close()
			plt.plot(nodes, marker="o") 
			plt.title(f"{dataset} Nodes")
			plt.savefig(f"{path}/{dataset}/{k}/nodes.png", dpi=100)
			plt.close()

			f = open(f"{path}/{dataset}/{k}/stats.txt", "w")
			f.write(f"\n{dataset}")
			f.write(f"\n\tLog Likelihood : {ll}")
			f.write(f"\n\tNodes : {nodes}")
			f.close()


			past3 = ll[-min(len(ll),3):]
				
			if n>=max_iter and round(np.std(past3), 3) <= 0.001:
				break
			
			i+=1
			n = min(n+step, max_iter)
			k1 += 1

		print("Log Likelihood",ll)
		print("Nodes",nodes)

		plt.close()
		# plot line 
		plt.plot(ll, marker="o") 
		#plt.show()
		plt.title(f"{dataset} Log Likelihood")
		plt.savefig(f"{path}/{dataset}/{k}/ll.png", dpi=100)
		plt.close()
		plt.plot(nodes, marker="o") 
		#plt.show()
		plt.title(f"{dataset} Nodes")
		plt.savefig(f"{path}/{dataset}/{k}/nodes.png", dpi=100)
		plt.close()
		

		lls.append(ll)
		nodes_k.append(nodes)
		k+=1

	plt.close()
	colors = ["aqua", "palegreen", "pink"]
	total_ll = np.zeros(max([len(lls[i]) for i in range(len(lls))]))
	for i in range(len(lls)):
		plt.plot(lls[i], marker="o", color =colors[i], label=(i+1))
		total_ll += np.array(lls[i])
	avg_ll = total_ll/len(lls)
	plt.plot(avg_ll, marker="o", color ="black", label="Mean")
	plt.title(f"{dataset} Log Likelihood")
	plt.legend()
	plt.savefig(f"{path}/{dataset}/ll.png", dpi=150)
	plt.close()

	total_nodes = np.zeros(max([len(nodes_k[i]) for i in range(len(nodes_k))]))
	for i in range(len(nodes_k)):
		plt.plot(nodes_k[i], marker="o", color =colors[i], label=(i+1))
		total_nodes += np.array(nodes_k[i])
	avg_nodes = total_nodes/len(nodes_k)
	plt.plot(avg_nodes, marker="o", color ="black", label="Mean")
	plt.title(f"{dataset} Nodes")
	plt.legend()
	plt.savefig(f"{path}/{dataset}/nodes.png", dpi=150)
	plt.close()

