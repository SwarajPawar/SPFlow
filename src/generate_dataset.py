

import numpy as np
import collections

import pandas as pd
import random
import csv

from spn.data.simulator import get_env
from spn.data.metaData import get_feature_names
import multiprocessing

instances = 500000
n_actions = 4
ext = ""


#datasets = ["Export_Textiles", 'Test_Strep', 'LungCancer_Staging', 'HIV_Screening', 'Computer_Diagnostician', 'Powerplant_Airpollution']
env = get_env('Export_Textiles', True)

def get_instance(id):
	state = env.reset()
	while True:
		action = random.randint(0, n_actions-1)
		state, reward, done = env.step(action)
		if done:
			instance = [id] + list(state[0])
			return instance

pool = multiprocessing.Pool()
batch = 10000


	
data = []

for i in range(int(instances/batch)):
	
	ids = [(i*batch)+j+1 for j in range(batch)]
	
	datablock = pool.map(get_instance, ids)
	data += datablock
	print((i+1)*batch)


fname = f"spn/data/{dataset}/{dataset}{ext}.tsv"


with open(fname, 'w') as file:
	wr = csv.writer(file,delimiter='\t')
	columns = ["ids"] + get_feature_names(dataset)
	wr.writerow(columns)
	for row in data:
			wr.writerow(row)


