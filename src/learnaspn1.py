'''

This Code is used to learn and evaluate
the SPN models for the given datasets
using the AnytimeSPN technique


'''

import numpy as np

from spn.algorithms.AnytimeSPN import AnytimeSPN

import logging
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings('ignore')


import pandas as pd
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType
from spn.io.Graphics import plot_spn
from os import path as pth
import sys, os
import time
import pickle
import matplotlib.pyplot as plt




#datasets = ["nltcs","msnbc", "plants", "kdd", "baudio", "jester", "bnetflix"]
datasets = ['wetgrass']
path = "wetgrass"



for dataset in datasets:
	
	print(f"\n\n\n{dataset}\n\n\n")
	
	#Get train and test datasets
	df = pd.read_csv(f"wetgrass/{dataset}.tsv", sep='\t')
	train = df.values

	test = train


	#Get dataset Context
	ds_context = Context(meta_types=[MetaType.DISCRETE]*train.shape[1])
	ds_context.add_domains(train)

	#Initialize ASPN
	aspn = AnytimeSPN(dataset, path, ds_context)
	#spn_structure, stats = aspn.learn_aspn(train, test)

	

	#Start anytime learning
	for i, output in enumerate(aspn.anytime_learn_spn(train, test, get_stats = True)):

		spn, stats = output
		

		#Plot the spn
		plot_spn(spn, f'{path}/spn{i+1}.pdf')

		#Get stats
		runtime = stats["runtime"]
		avg_ll = stats["ll"]
		nodes = stats["nodes"]

		# plot the statistics
		plt.close()
		plt.plot(range(1,len(runtime)+1), runtime, marker="o", label="Anytime")
		plt.title(f"{dataset} Run Time (in seconds)")
		plt.xlabel("Iteration")
		plt.ylabel("Run Time")
		plt.legend()
		plt.savefig(f"{path}/runtime.png", dpi=100)
		plt.close()

		plt.close()
		plt.plot(range(1,len(avg_ll)+1), avg_ll, marker="o", label="Anytime")
		plt.title(f"{dataset} Log Likelihood")
		plt.xlabel("Iteration")
		plt.ylabel("Log Likelihood")
		plt.legend()
		plt.savefig(f"{path}/ll.png", dpi=100)
		plt.close()


		plt.plot(range(1,len(nodes)+1), nodes, marker="o", label="Anytime")
		plt.title(f"{dataset} Nodes")
		plt.xlabel("Iteration")
		plt.ylabel("# Nodes")
		plt.legend()
		plt.savefig(f"{path}/nodes.png", dpi=100)
		plt.close()

