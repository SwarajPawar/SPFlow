

import numpy as np

from spn.algorithms.ASPN import AnytimeSPN

from spn.algorithms.Statistics import get_structure_stats_dict
from spn.io.Graphics import plot_spn
from spn.data.domain_stats import get_original_stats, get_max_stats

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



datasets = ["nltcs", "msnbc", "kdd", "jester", "baudio", "bnetflix"]

path = "cross_new"

#kfolds = 3
kfold = KFold(n_splits=3, shuffle=True)

for dataset in datasets:
	
	original = get_original_stats(dataset)
	upper = get_max_stats(dataset)



	k_ll = list()
	k_nodes = list()
	k_runtime = list()

	for i in range(3):

		#read the saved stats
		f = open(f"{path}/{dataset}/{i+1}/stats.txt", "r")
		f.readline()

		n = f.readline()
		n = n[n.index("[")+1:-2]
		n = n.split(", ")
		n = [float(x) for x in n]
		k_nodes.append(n)

		
		ll = f.readline()
		ll = ll[ll.index("[")+1:-2]
		ll = ll.split(", ")
		ll = [float(x) for x in ll]
		k_ll.append(ll)

		

		r = f.readline()
		r = r[r.index("[")+1:-2]
		r = r.split(", ")
		r = [float(x) for x in r]
		k_runtime.append(r)

		#plot the profiles
		plt.plot(range(1,len(k_ll[i])+1), k_ll[i], marker="o")
		plt.title(f"{dataset} Log Likelihood")
		plt.xlabel("Iteration")
		plt.ylabel("Log Likelihood")
		plt.savefig(f"{path}/{dataset}/{i+1}/ll.png", dpi=150)
		plt.close()

		plt.plot(range(1,len(k_nodes[i])+1), k_nodes[i], marker="o")
		plt.title(f"{dataset} # Nodes")
		plt.xlabel("Iteration")
		plt.ylabel("Log Likelihood")
		plt.savefig(f"{path}/{dataset}/{i+1}/nodes.png", dpi=150)
		plt.close()

		plt.plot(range(1,len(k_runtime[i])+1), k_runtime[i], marker="o")
		plt.title(f"{dataset} Run Time (in seconds)")
		plt.xlabel("Iteration")
		plt.ylabel("Log Likelihood")
		plt.savefig(f"{path}/{dataset}/{i+1}/runtime.png", dpi=150)
		plt.close()
	

	#Plot the mean
	plt.close()
	colors = ["red", "blue", "green"]

	maxlen = max([len(k_ll[i]) for i in range(len(k_ll))])
	total_ll = np.zeros(min([len(k_ll[i]) for i in range(len(k_ll))]))
	upperll = [upper["ll"]] * maxlen
	plt.plot(range(1, maxlen+1), upperll, linestyle="dashed", color ="darkred", linewidth=3, label="Upper Limit")
	originalll = [original["ll"]] * maxlen
	plt.plot(range(1, maxlen+1), originalll, linestyle="dotted", color ="purple", linewidth=3, label="LearnSPN")
	for i in range(len(k_ll)):
		plt.plot(range(1,len(k_ll[i])+1), k_ll[i], marker=f"{i+1}", color =colors[i], label=(i+1))
		total_ll += np.array(k_ll[i][:len(total_ll)])
	avg_ll = total_ll/len(k_ll)
	plt.plot(range(1,len(avg_ll)+1), avg_ll, marker="o", color ="black", linewidth=3, label="Mean")
	plt.title(f"{dataset} Log Likelihood")
	plt.legend()
	plt.xlabel("Iteration")
	plt.ylabel("Log Likelihood")
	plt.savefig(f"{path}/{dataset}/ll_{dataset}.png", dpi=150)
	plt.close()
	
	
	total_nodes = np.zeros(min([len(k_nodes[i]) for i in range(len(k_nodes))]))
	uppern = [upper["nodes"]] * maxlen
	plt.plot(range(1, maxlen+1), uppern, linestyle="dashed", color ="darkred", linewidth=3, label="Upper Limit")
	originaln = [original["nodes"]] * maxlen
	plt.plot(range(1, maxlen+1), originaln, linestyle="dotted", color ="purple", linewidth=3, label="LearnSPN")
	for i in range(len(k_nodes)):
		plt.plot(range(1,len(k_nodes[i])+1), k_nodes[i], marker=f"{i+1}", color =colors[i], label=(i+1))
		total_nodes += np.array(k_nodes[i][:len(total_nodes)])
	avg_nodes = total_nodes/len(k_nodes)
	plt.plot(range(1,len(avg_nodes)+1), avg_nodes, marker="o", color ="black", linewidth=3, label="Mean")
	plt.title(f"{dataset} Nodes")
	plt.legend()
	plt.xlabel("Iteration")
	plt.ylabel("# Nodes")
	plt.savefig(f"{path}/{dataset}/nodes_{dataset}.png", dpi=150)
	plt.close()


	total_time = np.zeros(min([len(k_runtime[i]) for i in range(len(k_runtime))]))
	uppertime = [upper["runtime"]] * maxlen
	plt.plot(range(1, maxlen+1), uppertime, linestyle="dashed", color ="darkred", linewidth=3, label="Upper Limit")
	originaltime = [original["runtime"]] * maxlen
	plt.plot(range(1, maxlen+1), originaltime, linestyle="dotted", color ="purple", linewidth=3, label="LearnSPN")
	for i in range(len(k_runtime)):
		plt.plot(range(1,len(k_runtime[i])+1), k_runtime[i], marker=f"{i+1}", color =colors[i], label=(i+1))
		total_time += np.array(k_runtime[i][:len(total_time)])
	avg_time = total_time/len(k_runtime)
	plt.plot(range(1,len(avg_time)+1), avg_time, marker="o", color ="black", linewidth=3, label="Mean")
	plt.title(f"{dataset} Run Time (in seconds)")
	plt.legend()
	plt.xlabel("Iteration")
	plt.ylabel("Run Time")
	plt.savefig(f"{path}/{dataset}/runtime_{dataset}.png", dpi=150)
	plt.close()
	
