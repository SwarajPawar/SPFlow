'''

This Code is used to learn and evaluate
the SPN models for the given datasets
using the LearnSPN algorithm


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
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian

from spn.algorithms.LearningWrappers import learn_parametric, learn_classifier


import warnings
warnings.filterwarnings('ignore')



import pandas as pd
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType
from spn.algorithms.Statistics import get_structure_stats_dict
from spn.algorithms.EM import EM_optimization
import matplotlib.pyplot as plt
from os import path as pth
import sys, os
import time
import multiprocessing
import pickle
import random
from keras.datasets import mnist
from PIL import Image

#Initialize parameters

cols="rdc"
rows="kmeans"
min_instances_slice=200
threshold=0.3
ohe=False
leaves = create_histogram_leaf
rand_gen=None
cpus=-1


path = "mnist"

dataset = "mnist"



(x_train, y_train), (x_test, y_test) = mnist.load_data()


#Create output directory 
print(f"\n\n\n{dataset}\n\n\n")
if not pth.exists(f'{path}/models'):
	try:
		os.makedirs(f'{path}/models')
	except OSError:
		print ("Creation of the directory %s failed" % path)
		sys.exit()



valid = list()



for i, x in enumerate(x_test):
	img = Image.fromarray(x, mode='L')
	img = img.resize((10,10))
	x = np.asarray(img)
	x = [y_test[i]] + list(np.reshape(x, (x.shape[0]*x.shape[1])))
	valid.append(x)

test = list()

for i, x in enumerate(x_train):
	img = Image.fromarray(x, mode='L')
	img = img.resize((10,10))
	x = np.asarray(img)
	x = [y_train[i]] + list(np.reshape(x, (x.shape[0]*x.shape[1])))
	valid.append(x)

for i, x in enumerate(x_test):
	img = Image.fromarray(x, mode='L')
	img = img.resize((10,10))
	x = np.asarray(img)
	x = [np.nan] + list(np.reshape(x, (x.shape[0]*x.shape[1])))
	test.append(x)


valid = np.array(valid)
print(valid)
print(valid.shape)

test = np.array(test)
print(test)
print(test.shape)

	

'''
ds_context = Context(parametric_types=[Categorical]+ [Gaussian]*(test.shape[1]-1))
ds_context.add_domains(test)
'''

file = open(f"{path}/models/spn_{dataset}.pkle","rb")
spn = pickle.load(file)
file.close()

EM_optimization(spn, valid)
print("Optimized")

file = open(f"{path}/models/spn_{dataset}_EM.pkle",'wb')
pickle.dump(spn, file)
file.close()

nodes = get_structure_stats_dict(spn)["nodes"]
parameters = get_structure_stats_dict(spn)["parameters"]
layers = get_structure_stats_dict(spn)["layers"]


print(f"\n\tNodes : {nodes}")
print(f"\n\tParameters : {parameters}")
print(f"\n\tLayers : {layers}")

from spn.algorithms.MPE import mpe


results = mpe(spn, test)

pred = list(results[:,0])
true = list(y_test)


from sklearn import metrics

report = metrics.classification_report(true, pred)
print(f'\n\nReport : \n{report}')

prfs = metrics.precision_recall_fscore_support(true, pred)
prfs_micro = metrics.precision_recall_fscore_support(true, pred, average='micro')
cm = metrics.confusion_matrix(true, pred)

print(f"\n\t{prfs}")
print(f"\n\t{prfs_micro}")
print(f"\n\tConfusion Matrix : {cm}")

f = open(f"{path}/{dataset}/statsnew.txt", "w")
f.write(f"\n{dataset}")
f.write(f"\n\tNodes : {nodes}")
f.write(f"\n\tParameters : {parameters}")
f.write(f"\n\tLayers : {layers}")
f.write(f"\n\tReport : \n{report}")
f.write(f"\n\tConfusion Matrix : {cm}")
f.write(f"\n\t{prfs}")
f.write(f"\n\t{prfs_micro}")
f.close()


