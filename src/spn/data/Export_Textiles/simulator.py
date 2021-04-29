

import numpy as np
import collections

import pandas as pd
import random

class Export_Textiles:

	'''
	State:
	0: 'Same'
	1: 'Severely_bad'
	2: 'Slightly_worse'

	Action:
	0: 'Now'
	1: 'After_6_mos'
	2: 'After_12_mos'
	'''


	def __init__(self):
		
		#rewards[Economical_State][Export_Decision]

		self.rewards = {1: 
							{1: 900500.0, 
							0: -711000.0, 
							2: 766000.0}, 
						0:
							{0: 2726000.0, 
							2: 210000.0, 
							1: 2357000.0},
						2:
							{ 1: 1694500.0,
							0: 1870000.0, 
							2: 1425000.0}
						}
						
	def reset(self):

		self.Economical_State = np.nan
		self.Export_Decision = np.nan
		self.Profit = np.nan


		return self.state()
			
	def step(self, action):
		
		self.Export_Decision = action

		p = random.random()
		if p<0.4:
			self.Economical_State = 0
		elif p<0.7:
			self.Economical_State = 1
		else:
			self.Economical_State = 2

		self.Profit = self.rewards[self.Economical_State][self.Export_Decision]

		return self.state(), self.Profit, True

	def state(self):
		return [[self.Economical_State, self.Export_Decision, self.Profit]]

