

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
from spn.algorithms.MEU import meu
from spn.io.Graphics import plot_spn
import matplotlib.pyplot as plt
from os import path as pth
import sys, os
import pickle


datasets = ['Powerplant_Airpollution']
path = "LearnSPMN_results"






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
	
	

	

	#Get the model from the file
	file = open(f"{plot_path}/spmn_original.pkle","rb")
	spmn = pickle.load(file)
	file.close()

	test_data = [[1, 1, 0, 0, np.nan]]
	meus = meu(spmn, test_data)
	print(meus)


		
