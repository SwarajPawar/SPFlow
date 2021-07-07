

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
ext = "_new"

dataset = 'FrozenLake'
#datasets = ["Export_Textiles", 'Test_Strep', 'LungCancer_Staging', 'HIV_Screening', 'Computer_Diagnostician', 'Powerplant_Airpollution']
env = get_env('FrozenLake', True)

def get_instance(id):
	state = env.reset()
	while True:
		action = random.randint(0, n_actions-1)
		state, reward, done = env.step(action)
		if done:
			instance = [id] + list(state[0])
			return instance

'''
def get_action(state):
	if state==0:
		p=random.random()
		if p<0.5:
			return 1
		else:
			return 2
	if state==1:
		return 2
	if state==2:
		return 1
	if state==3:
		return 0
	if state==4:
		return 1
	if state==5:
		return random.randint(0, n_actions-1)
	if state==6:
		return 1
	if state==7:
		return random.randint(0, n_actions-1)
	if state==8:
		return 2
	if state==9:
		p=random.random()
		if p<0.5:
			return 1
		else:
			return 2
	if state==10:
		return 1
	if state==11:
		return random.randint(0, n_actions-1)
	if state==12:
		return random.randint(0, n_actions-1)
	if state==13:
		return 2
	if state==14:
		return 2
	if state==15:
		return 2

def get_instance1(id):
	state = env.reset()
	k=0
	while True:
		action = get_action(state[0][k*2])
		state, reward, done = env.step(action)
		k+=1
		if done:
			instance = [id] + list(state[0])
			return instance
'''

pool = multiprocessing.Pool()
batch = 10000


	
data = []

for i in range(int(instances/batch)):
	
	ids = [(i*batch)+j+1 for j in range(batch)]
	'''
	if i < 6:
		datablock = datablock = pool.map(get_instance1, ids)
		data += datablock
	else:
	'''
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


