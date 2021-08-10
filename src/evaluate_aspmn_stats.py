

import numpy as np

import logging
logger = logging.getLogger(__name__)


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from os import path as pth
import sys, os
import random

from sklearn.model_selection import train_test_split
from spn.data.metaData import get_partial_order, get_utilityNode, get_decNode, get_feature_names, get_feature_labels
from spn.data.domain_stats import get_original_stats
from spn.algorithms.Statistics import get_structure_stats_dict
from spn.structure.StatisticalTypes import MetaType
from spn.algorithms.SPMNDataUtil import align_data
from spn.algorithms.SPMN import SPMN
from spn.algorithms.ASPMN import Anytime_SPMN
from spn.io.Graphics import plot_spn
import matplotlib.pyplot as plt
from os import path as pth
import sys, os
import pickle


datasets = ['Export_Textiles', 'Powerplant_Airpollution', 'HIV_Screening', 'Computer_Diagnostician', 'Test_Strep', 'LungCancer_Staging']
datasets = ['Navigation']
path = "new_results_depth"

model_count = 11





for dataset in datasets:
	
	print(f"\n\n\n{dataset}\n\n\n")
	plot_path = f'{path}/{dataset}'

	

	#Get all the parameters
	partial_order = get_partial_order(dataset)
	utility_node = get_utilityNode(dataset)
	decision_nodes = get_decNode(dataset)
	feature_names = get_feature_names(dataset)
	feature_labels = get_feature_labels(dataset)
	meta_types = [MetaType.DISCRETE]*(len(feature_names)-1)+[MetaType.UTILITY]

	#Get Baseline stats
	#original_stats = get_original_stats(dataset)
	
	all_edges = list()
	all_layers = list()

	#Initialize anytime Learning
	aspmn = Anytime_SPMN(dataset, path, partial_order , decision_nodes, utility_node, feature_names, feature_labels, meta_types, cluster_by_curr_information_set=True, util_to_bin = False)
	
	#Start evaluation
	for model in range(model_count):

		#Get the model from the file
		file = open(f"{plot_path}/models/spmn_{model+1}.pkle","rb")
		spmn = pickle.load(file)
		file.close()

		#Get edges and layers for the SPMNs
		struct_stats = aspmn.evaluate_structure_stats(spmn = spmn)
		all_edges.append(struct_stats['edges'])
		all_layers.append(struct_stats['layers'])

		#Save the results to a file
		f = open(f"{plot_path}/struct_stats.txt", "w")
		f.write(f"\n\t#Edges : {all_edges}")
		f.write(f"\n\t#Layers : {all_layers}")
		f.close()

		#Plots
		plt.close()
		plt.plot(range(1,len(all_edges)+1), all_edges, marker="o", label="Anytime")
		plt.title(f"{dataset} #Edges")
		plt.legend()
		plt.savefig(f"{plot_path}/edges.png", dpi=100)
		plt.close()

		plt.close()
		plt.plot(range(1,len(all_layers)+1), all_layers, marker="o", label="Anytime")
		plt.title(f"{dataset} #Layers")
		plt.legend()
		plt.savefig(f"{plot_path}/layers.png", dpi=100)
		plt.close()
		

		
