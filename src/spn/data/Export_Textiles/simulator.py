

import numpy as np
import collections

import pandas as pd
import random

class Export_Textiles:

	'''
	----------------------------------------------
	Variables and their encoded values:
	----------------------------------------------

	Economical_State:
	0: 'Same'
	1: 'Severely_bad'
	2: 'Slightly_worse'


	----------------------------------------------


	Export_Decision:
	0: 'Now'
	1: 'After_6_mos'
	2: 'After_12_mos'


	-----------------------------------------------
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
					
		#Done Indicator
		self.done = False


	#Initialize the variables to np.nan	
	def reset(self):

		self.Economical_State = np.nan
		self.Export_Decision = np.nan
		self.Profit = np.nan
		self.done = False


		return self.state()
			

	#Perform the action and get observed variables as per the CPTs		
	def step(self, action):
		
		self.Export_Decision = action

		p = random.random()
		if p<0.4:
			self.Economical_State = 0
		elif p<0.7:
			self.Economical_State = 1
		else:
			self.Economical_State = 2

		#Return the reward if all actions are done
		self.Profit = self.rewards[self.Economical_State][self.Export_Decision]

		self.done = True

		return self.state(), self.Profit, self.done

	#Return the state as given by the partial order
	def state(self):
		if not self.done:
			return [[np.nan, self.Export_Decision, np.nan]]
		else:
			return [[self.Economical_State, self.Export_Decision, self.Profit]]
