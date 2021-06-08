

import numpy as np
import collections

import pandas as pd
import random
import csv

from spn.data.simulator import get_env
from spn.data.metaData import get_feature_names

instances = 100000
n_actions = 2
ext = "_new"

datasets = ['Computer_Diagnostician_v2']
#datasets = ["Export_Textiles", 'Test_Strep', 'LungCancer_Staging', 'HIV_Screening', 'Computer_Diagnostician', 'Powerplant_Airpollution']

for dataset in datasets:

	env = get_env(dataset)
	data = []

	for i in range(instances):

		if (i+1)%1000 == 0:
			print(i+1)
		state = env.reset()
		while True:
			action = random.randint(0, n_actions-1)
			state, reward, done = env.step(action)
			if done:
				instance = [i+1] + list(state[0])
				data.append(instance)
				break


	fname = f"spn/data/{dataset}/{dataset}{ext}.tsv"
	

	with open(fname, 'w') as file:
		wr = csv.writer(file,delimiter='\t')
		columns = ["ids"] + get_feature_names(dataset)
		wr.writerow(columns)
		for row in data:
				wr.writerow(row)


