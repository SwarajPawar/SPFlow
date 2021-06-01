

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
from spn.algorithms.ASPMN2 import Anytime_SPMN
import matplotlib.pyplot as plt
from os import path as pth
import sys, os


datasets = ['HIV_Screening', 'Computer_Diagnostician', 'Powerplant_Airpollution', 'LungCancer_Staging', 'Export_Textiles', 'Test_Strep']
datasets = ['Export_Textiles']
path = "test"
path = "improve2"


for dataset in datasets:
	
	print(f"\n\n\n{dataset}\n\n\n")


	partial_order = get_partial_order(dataset)
	utility_node = get_utilityNode(dataset)
	decision_nodes = get_decNode(dataset)
	feature_names = get_feature_names(dataset)
	feature_labels = get_feature_labels(dataset)
	meta_types = [MetaType.DISCRETE]*(len(feature_names)-1)+[MetaType.UTILITY]

			
	df = pd.read_csv(f"spn/data/{dataset}/{dataset}_new.tsv", sep='\t')

	df, column_titles = align_data(df, partial_order)

	data = df.values
	#train, test = train_test_split(data, test_size=0.9, shuffle=True)
	test_size = int(data.shape[0]*0.2)
	train, test = data, np.array(random.sample(list(data), test_size))


	
	#train, test = data[:int(data.shape[0]*0.7)], data[int(data.shape[0]*0.7):]
	#print(train.shape)
	#print(test.shape)

	
	aspmn = Anytime_SPMN(dataset, path, partial_order , decision_nodes, utility_node, feature_names, feature_labels, meta_types, cluster_by_curr_information_set=True, util_to_bin = False)
	aspmn.learn_aspmn(train, test)
