

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

logger = logging.getLogger(__name__)


import warnings
warnings.filterwarnings('ignore')



import pandas as pd
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType
import matplotlib.pyplot as plt
from os import path as pth
import sys, os


min_instances_slice=200
threshold=0.3
ohe=False
leaves = create_histogram_leaf
rand_gen=None
cpus=-1


#datasets = ["nltcs","msnbc", "plants", "kdd", "baudio", "jester", "bnetflix"]
datasets = ["msnbc", "kdd"]
path = "cross1"

kfold = KFold(n_splits=5)

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
	data = df.values
	print(data.shape)
	df2 = pd.read_csv(f"spn/data/binary/{dataset}.test.data", sep=',')
	data2 = df2.values
	print(data2.shape)
	#data = np.concatenate((data1, data2))





	max_iter = data.shape[1]
	rows, var = data.shape
	ds_context = Context(meta_types=[MetaType.DISCRETE]*var)
	ds_context.add_domains(data)

	lls = list()
	nodes_k = list()
	
	i = 0
	for traini, testi in kfold.split(data):

		i+=1
		if i>3:
			break 

		train, test = data[traini], data[testi]
		#test = random.sample(list(test), 2000)
		plot_path = f"{path}/{dataset}/{i}"
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

		k = 0
		while True:
			split_cols = get_split_cols_distributed_RDC_py(rand_gen=rand_gen, ohe=ohe, n_jobs=cpus, n=round(n))
			split_rows = get_split_rows_XMeans(limit=k1, returnk=False)
			nextop = get_next_operation(min_instances_slice)

			spn = learn_structure(train, ds_context, split_rows, split_cols, leaves, nextop)

			nodes.append(get_structure_stats_dict(spn)["nodes"])
			from spn.io.Graphics import plot_spn

			plot_spn(spn, f'{path}/{dataset}/{i}/spn{k}.png')

			from spn.algorithms.Inference import log_likelihood
			total_ll = 0
			for instance in test:
				import numpy as np
				test_data = np.array(instance).reshape(-1, var)
				total_ll += log_likelihood(spn, test_data)[0][0]
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
			print(nodes[k])
			print(ll[k])
			print(ll)
			print(nodes)
			print("\n\n\n\n\n")
			
			k+=1
			
			plt.close()
			# plot line 
			plt.plot(ll, marker="o") 
			plt.title(f"{dataset} Log Likelihood")
			plt.savefig(f"{path}/{dataset}/{i}/ll.png", dpi=100)
			plt.close()
			plt.plot(nodes, marker="o") 
			plt.title(f"{dataset} Nodes")
			plt.savefig(f"{path}/{dataset}/{i}/nodes.png", dpi=100)
			plt.close()
			
			
			n = min(n+step, max_iter)
			k1 += 1

		print("Log Likelihood",ll)
		print("Nodes",nodes)

		plt.close()
		# plot line 
		plt.plot(ll, marker="o") 
		#plt.show()
		plt.title(f"{dataset} Log Likelihood")
		plt.savefig(f"{path}/{dataset}/{i}/ll.png", dpi=100)
		plt.close()
		plt.plot(nodes, marker="o") 
		#plt.show()
		plt.title(f"{dataset} Nodes")
		plt.savefig(f"{path}/{dataset}/{i}/nodes.png", dpi=100)
		plt.close()

		lls.append(ll)
		nodes_k.append(nodes)

	plt.close()
	colors = ["aqua", "palegreen", "pink"]
	total_ll = np.zeroes(max([len(lls[i]) for i in range(len(lls))]))
	for i in range(len(lls)):
		plt.plot(lls[i], marker="o", color =colors[i], label=(i+1))
		total_ll += np.array(lls[i])
	avg_ll = total_ll/len(lls)
	plt.plot(avg_ll, marker="o", color ="black", label="Mean")
	plt.title(f"{dataset} Log Likelihood")
	plt.legend()
	plt.savefig(f"{path}/{dataset}/ll.png", dpi=150)
	plt.close()

	total_nodes = np.zeroes(max([len(nodes_k[i]) for i in range(len(nodes_k))]))
	for i in range(len(nodes_k)):
		plt.plot(nodes_k[i], marker="o", color =colors[i], label=(i+1))
		total_nodes += np.array(nodes_k[i])
	avg_nodes = total_nodes/len(nodes_k)
	plt.plot(avg_nodes, marker="o", color ="black", label="Mean")
	plt.title(f"{dataset} Nodes")
	plt.legend()
	plt.savefig(f"{path}/{dataset}/nodes.png", dpi=150)
	plt.close()



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






import pandas as pd
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType

dataset = "nltcs"

cols="rdc"
rows="kmeans"
min_instances_slice=200
threshold=0.3
ohe=False
leaves = create_histogram_leaf
rand_gen=None
cpus=-1

df = pd.read_csv(f"spn/data/binary/{dataset}.ts.data", sep=',')
data = df.values
print(data.shape)
max_iter = data.shape[1]
samples, var = data.shape
ds_context = Context(meta_types=[MetaType.DISCRETE]*var)
ds_context.add_domains(data)

df2 = pd.read_csv(f"spn/data/binary/{dataset}.test.data", sep=',')
test = df2.values

ll = list()
nodes = list()
k1 = [i for i in range(1,max_iter+1)]
n = [i for i in range(int(max_iter**0.5),max_iter+1)]

i,j,k = 0,0,0
while True:
		
	split_cols=get_split_cols_distributed_RDC_py(rand_gen=rand_gen, ohe=ohe, n_jobs=cpus, n=n[j])
	split_rows = get_split_rows_XMeans(n=k1[i])
	nextop = get_next_operation(min_instances_slice)

	spn = learn_structure(data, ds_context, split_rows, split_cols, leaves, nextop)

	nodes.append(get_structure_stats_dict(spn)["nodes"])
	#from spn.io.Graphics import plot_spn

	#plot_spn(spn, 'basicspn'+str(k)+'.png')

	from spn.algorithms.Inference import log_likelihood
	total_ll = 0
	for instance in test:
		import numpy as np
		test_data = np.array(instance).reshape(-1, var)
		total_ll += log_likelihood(spn, test_data)[0][0]
	ll.append(total_ll/len(test))
	
	if n[j]==max_iter:
		break
	print("\n\n\n\n\n")
	print(k1[i],n[j])
	print(nodes[k])
	print(ll[k])
	print(ll)
	print(nodes)
	print("\n\n\n\n\n")
	k+=1
	if i<len(k1)-1:
		i+=1
	if j<len(k1)-1:
		j+=1

import matplotlib.pyplot as plt 

print(ll)
print(nodes)
# plot line 
plt.plot(ll) 
plt.show()
plt.plot(nodes) 
plt.show()



'''

