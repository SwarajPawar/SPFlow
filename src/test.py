

import numpy as np
import collections

import pandas as pd
import random
import csv


import matplotlib.pyplot as plt

from spn.data.simulator import get_env
import multiprocessing

def get_reward(ids):

	state = env.reset()
	while(True):
		action = 0
		state, reward, done = env.step(action)
		if done:
			return reward

dataset = "Computer_Diagnostician_v2"
env = get_env(dataset)
total_reward = 0
#trials = 200000
batch_count = 50
batch_size = 20000 #int(trials / batch_count)
batch = list()

pool = multiprocessing.Pool()

for z in range(batch_count):
	
	ids = [None for x in range(batch_size)]
	rewards = pool.map(get_reward, ids)
	
	batch.append(sum(rewards)/batch_size)
	printProgressBar(z+1, batch_count, prefix = f'Average Reward Evaluation :', suffix = 'Complete', length = 50)
	print(f"{np.mean(batch)} \t {np.std(batch)}")

avg_rewards = np.mean(batch)
reward_dev = np.std(batch)