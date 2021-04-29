

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
from spn.data.metaData import *
from spn.structure.StatisticalTypes import MetaType
from spn.algorithms.SPMNDataUtil import align_data
from spn.algorithms.SPMN import SPMN
from spn.algorithms.MEUTopDown import spmn_topdowntraversal_and_bestdecisions
import matplotlib.pyplot as plt
from os import path as pth
import sys, os

datasets = ["Export_Textiles"]
#datasets = [f"Dataset{i+1}" for i in range(6)]

path = "test_dec"



for dataset in datasets:
	
	print(f"\n\n\n{dataset}\n\n\n")
	plot_path = f"{path}/{dataset}"
	if not pth.exists(plot_path):
		try:
			os.makedirs(plot_path)
		except OSError:
			print ("Creation of the directory %s failed" % plot_path)
			sys.exit()


	partial_order = get_partial_order(dataset)
	utility_node = get_utilityNode(dataset)
	decision_nodes = get_decNode(dataset)
	feature_names = get_feature_names(dataset)
	feature_labels = get_feature_labels(dataset)
	meta_types = [MetaType.DISCRETE]*(len(feature_names)-1)+[MetaType.UTILITY]

			
	df = pd.read_csv(f"spn/data/{dataset}/{dataset}.tsv", sep='\t')

	df, column_titles = align_data(df, partial_order)  # aligns data in partial order sequence
	'''
	col_ind = column_titles.index(utility_node[0]) 
	df_without_utility = df1.drop(df1.columns[col_ind], axis=1)
	from sklearn.preprocessing import LabelEncoder
	# transform categorical string values to categorical numerical values
	df_without_utility_categorical = df_without_utility.apply(LabelEncoder().fit_transform)  
	df_utility = df1.iloc[:, col_ind]
	df = pd.concat([df_without_utility_categorical, df_utility], axis=1, sort=False)
	'''
	data = df.values
	train, test = train_test_split(data, test_size=0.2, shuffle=True)

	
	spmn = SPMN(partial_order , decision_nodes, utility_node, feature_names, meta_types, cluster_by_curr_information_set = True, util_to_bin = False)
	spmn = spmn.learn_spmn(train)
	print("Done")
	
	
	nodes = get_structure_stats_dict(spmn)["nodes"]
    
    plot_spn(spmn, f'{path}/{dataset}/spmn.pdf', feature_labels=feature_labels)



    '''
	total_ll = 0
	for instance in test:
		test_data = np.array(instance).reshape(-1, len(feature_names))
		total_ll += log_likelihood(spmn, test_data)[0][0]
	ll = (total_ll/len(test))


	test_data = [[np.nan]*len(feature_names)]
	m = meu(spmn, test_data)
	meus = (m[0])

	f = open(f"{path}/{dataset}/stats.txt", "w")
	f.write(f"\n{dataset}")
	f.write(f"\n\tLog Likelihood : {ll}")
	f.write(f"\n\tMEU : {meus}")
	f.write(f"\n\tNodes : {nodes}")
	f.close()
	'''

	from spn.data.Export_Textiles.simulator import ExportTextiles

	env = ExportTextiles()
	test_data = [[0, np.nan, np.nan]]
	output = spmn_topdowntraversal_and_bestdecisions(spmn, test_data)
	print(output)
	test_data = [[1, np.nan, np.nan]]
	output = spmn_topdowntraversal_and_bestdecisions(spmn, test_data)
	print(output)
	test_data = [[2, np.nan, np.nan]]
	output = spmn_topdowntraversal_and_bestdecisions(spmn, test_data)
	print(output)