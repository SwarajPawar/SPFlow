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

		

train = list()
for i, x in enumerate(x_train):
	img = Image.fromarray(x, mode='L')
	img = img.resize((10,10))
	x = np.asarray(img)
	x = [y_train[i]] + list(np.reshape(x, (x.shape[0]*x.shape[1])))
	train.append(x)

train = np.array(train)
print(train)
print(train.shape)


ds_context = Context(parametric_types=[Categorical]+ [Gaussian]*(train.shape[1]-1))
ds_context.add_domains(train)

print("\n\nLearning SPN")
start = time.time()
spn = learn_classifier(train, ds_context, learn_parametric, 0)
end = time.time()
print("\nSPN Learned!")

print("\n\nRun Time: ", (end-start))

file = open(f"{path}/models/spn_{dataset}.pkle",'wb')
pickle.dump(spn, file)
file.close()
'''
from spn.io.Graphics import plot_spn
#Plot spn
plot_spn(spn, f'{path}/{dataset}_spn.pdf')
'''
'''
test = [[1,0,0,0,np.nan],
		[0,0,1,0,np.nan]]

test = np.array(test)

from spn.algorithms.MPE import mpe
print(mpe(spn, test))

'''
