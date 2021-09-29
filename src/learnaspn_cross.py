

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



datasets = ["jester", "bnetflix", "kdd", "plants"]
#datasets = ["msnbc"]
path = "cross_new1"

#kfolds = 3
kfold = KFold(n_splits=3, shuffle=True)

for dataset in datasets:
	
	print(f"\n\n\n{dataset}\n\n\n")
	
			
	df = pd.read_csv(f"spn/data/binary/{dataset}.ts.data", sep=',')
	data1 = df.values
	print(data1.shape)
	df2 = pd.read_csv(f"spn/data/binary/{dataset}.test.data", sep=',')
	data2 = df2.values
	print(data2.shape)
	data = np.concatenate((data1, data2))
	print(data.shape)

	original = get_original_stats(dataset)
	upper = get_max_stats(dataset)


	k_ll = list()
	k_nodes = list()
	k_runtime = list()
	

	k = 1
	for trainidx, testidx in kfold.split(data):
		#train, test = train_test_split(data, test_size=0.3, shuffle=True)
		
		train, test = data[trainidx], data[testidx]
		#test = np.array(random.sample(list(test), 5000))

		plot_path = f"{path}/{dataset}/{k}"
		if not pth.exists(plot_path):
			try:
				os.makedirs(plot_path)
			except OSError:
				print ("Creation of the directory %s failed" % plot_path)
				sys.exit()

		#Get dataset Context
		ds_context = Context(meta_types=[MetaType.DISCRETE]*train.shape[1])
		ds_context.add_domains(train)

		#Initialize ASPN
		aspn = AnytimeSPN(dataset, path, ds_context, k)
		#spn_structure, stats = aspn.learn_aspn(train, test)

		ll = list()
		nodes = list()
		runtime = list()		

		#Start anytime learning
		for i, output in enumerate(aspn.learn_aspn(train, test, get_stats = True)):

			spn, stats = output
			

			#Plot the spn
			plot_spn(spn, f'{plot_path}/spn{i+1}.pdf')

			#Get stats
			runtime = stats["runtime"]
			ll = stats["ll"]
			nodes = stats["nodes"]
			
			# plot the statistics
			plt.close()
			plt.plot(range(1,len(runtime)+1), runtime, marker="o", label="Anytime")
			plt.title(f"{dataset} Run Time (in seconds)")
			plt.xlabel("Iteration")
			plt.ylabel("Run Time")
			plt.legend()
			plt.savefig(f"{plot_path}/runtime.png", dpi=100)
			plt.close()

			plt.close()
			plt.plot(range(1,len(ll)+1), ll, marker="o", label="Anytime")
			plt.title(f"{dataset} Log Likelihood")
			plt.xlabel("Iteration")
			plt.ylabel("Log Likelihood")
			plt.legend()
			plt.savefig(f"{plot_path}/ll.png", dpi=100)
			plt.close()


			plt.plot(range(1,len(nodes)+1), nodes, marker="o", label="Anytime")
			plt.title(f"{dataset} Nodes")
			plt.xlabel("Iteration")
			plt.ylabel("# Nodes")
			plt.legend()
			plt.savefig(f"{plot_path}/nodes.png", dpi=100)
			plt.close()

		k_ll.append(ll)
		k_runtime.append(runtime)
		k_nodes.append(nodes)

		k+=1

	
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
	plt.savefig(f"{path}/{dataset}/ll.png", dpi=150)
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
	plt.savefig(f"{path}/{dataset}/nodes.png", dpi=150)
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
	plt.savefig(f"{path}/{dataset}/runtime.png", dpi=150)
	plt.close()
	
