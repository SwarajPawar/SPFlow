

import numpy as np
import collections

import pandas as pd
import random
import csv


import matplotlib.pyplot as plt


import threading
import time

from spn.data.simulator import get_env
import multiprocessing



class simThread (threading.Thread):

	def __init__(self, id, env):
		threading.Thread.__init__(self)
		self.id = id
		self.env = env


	def run(self):
		state = env.reset()
		while(True):
			output = random.randint(1,2)#best_next_decision(spmn, state)
			action = output
			state, reward, done = env.step(action)
			if done:
				rewards[self.id] = reward
				break

dataset = 'Export_Textiles'
env = get_env(dataset)

def get_reward():
	state = env.reset()
	while(True):
		output = random.randint(1,2)#best_next_decision(spmn, state)
		action = output
		state, reward, done = env.step(action)
		if done:
			return reward




total_reward = 0
reward_all = list()

trials = 10
pool = multiprocessing.Pool(10)

start = time.time()
for z in range(trials):

	'''
	threads = list()
	rewards = [None for i in range(10000)]
	
	for i in range(10000):
		thread = simThread(i, env)
		thread.start()

	for t in threads:
		t.join()
	'''
	rewards = zip(*pool.map(get_reward))
	reward_all += rewards
end = time.time()

print(end - start)
#print(reward_all)


			

rewards = list()
trials = 10000
start = time.time()
for z in range(trials):

	rewards.append(get_reward())
end = time.time()
print(end - start)


#seq : 0.0003960132598876953
	
		
		

		

	
