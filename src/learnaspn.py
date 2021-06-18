'''

This Code is used to learn and evaluate
the SPN models for the given datasets
using the AnytimeSPN technique


'''

import numpy as np

from spn.algorithms.ASPN import AnytimeSPN

import logging

logger = logging.getLogger(__name__)


import warnings

warnings.filterwarnings('ignore')



import pandas as pd
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType
from os import path as pth
import sys, os
import time
import pickle




datasets = ["nltcs","msnbc", "plants", "kdd", "baudio", "jester", "bnetflix"]
datasets = ['nltcs']
path = "output"



for dataset in datasets:
	
	print(f"\n\n\n{dataset}\n\n\n")
	
	#Get train and test datasets
	df = pd.read_csv(f"spn/data/binary/{dataset}.ts.data", sep=',')
	train = df.values

	df2 = pd.read_csv(f"spn/data/binary/{dataset}.test.data", sep=',')
	test = df2.values

	#Get dataset Context
	ds_context = Context(meta_types=[MetaType.DISCRETE]*train.shape[1])
	ds_context.add_domains(train)

	#Initialize ASPN
	aspn = AnytimeSPN(dataset, path, ds_context)
	spn_structure, stats = aspn.learn_aspn(train, test)
	#Start anytime learning
	for i, output in enumerate(aspn.learn_aspn(train, test)):

		spn, stats = output
		#Save models
		if not pth.exists(f"{path}/{dataset}/models"):
			try:
				os.makedirs(f"{path}/{dataset}/models")
			except OSError:
				print ("Creation of the directory failed")
				sys.exit()

		file = open(f"{path}/{dataset}/models/spn_{i}.pkle",'wb')
		pickle.dump(spn, file)
		file.close()
