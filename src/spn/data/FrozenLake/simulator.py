

import numpy as np
from collections import Counter

import pandas as pd
import random
import gym



class FrozenLake:

	'''
	---------------------------------------------------
	Variables and their encoded values:
	---------------------------------------------------



	State: 

	0 to 15 grid locations


	-----------------------------------------------------

	Actions:

	0: Left
	1: Down
	2: Right
	3: Up


	----------------------------------------------------


	'''

	def __init__(self):
		
		self.tot_dec = 10
		self.env = gym.make('FrozenLake-v0')
		#Done Indicator
		self.done = False
			

	#Initialize the variables to np.nan				
	def reset(self):
		
		self.locs = [np.nan]*11
		self.cur_loc = self.env.reset()
		self.locs[0] = self.cur_loc

		self.decs = [np.nan]*10

		self.reward = 0
		self.cur_dec = 0

		self.done = False
		self.terminal = False
		
		return self.state()


	#Perform the action and get observed variables as per the CPTs		
	def step(self, action):

		self.decs[self.cur_dec] = action
		self.cur_dec += 1

		if not self.terminal:
			self.cur_loc, cur_reward, self.teminal, __ =self.env.step(action)
			self.reward += cur_reward

		self.locs[self.cur_dec] = self.cur_loc

		if self.cur_dec == self.tot_dec:
			self.done = True

		return self.state(), self.reward, self.done


	#Return the state as given by the partial order
	def state(self):
		
		if not self.done:
			seq = list()
			for i in range(self.tot_dec):
				seq += [np.nan, self.decs[i]]
			return [seq + [np.nan]]
		else:
			seq = list()
			for i in range(self.tot_dec):
				seq += [self.locs[i], self.decs[i]]
			seq += [self.locs[-1]]
			return [seq + [self.reward]]



