

import numpy as np
import collections

import pandas as pd
import random
import csv

from spn.data.simulator import get_env
from spn.data.metaData import get_feature_names



dataset = "LungCancer_Staging"
datasets = ['Test_Strep', 'LungCancer_Staging', 'HIV_Screening', 'Computer_Diagnostician', 'Powerplant_Airpollution']

data = pd.read_csv(f"spn/data/{dataset}/{dataset}_new.tsv", delimiter = "\t")
	
data = data.values[:,1:]
#data = data[:,4:]

unique = []

for x in data:
	z = tuple((x[1], x[6], x[5], x[7], x[8]))
	if z not in unique:
		unique.append(z)

for u in unique:
	print(u)

print(len(unique))