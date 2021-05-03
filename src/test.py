

import numpy as np
import collections

import pandas as pd
import random
import csv

from spn.data.simulator import get_env
from spn.data.metaData import get_feature_names



dataset = "HIV_Screening"
datasets = ['Export_Textiles', 'Test_Strep', 'LungCancer_Staging', 'HIV_Screening', 'Computer_Diagnostician', 'Powerplant_Airpollution']

data = pd.read_csv(f"spn/data/{dataset}/{dataset}_new1.tsv", delimiter = "\t")
	
data = data.values[:,1:]
#data = data[:,3:]

unique = []

for x in data:
	z = tuple((x[1], x[4], x[5], x[6]))
	#z = tuple(x)
	if z not in unique:
		unique.append(z)

for u in unique:
	print(u)

print(len(unique))