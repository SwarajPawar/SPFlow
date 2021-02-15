"""
Created on March 30, 2018

@author: Alejandro Molina
"""

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






























