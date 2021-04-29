

import numpy as np
from collections import Counter

import pandas as pd
import random




class Powerplant_Airpollution:
	'''

	State: 

	Coal_Worker_Strike
	0: No
	1: Yes

	Strike_Resolution
	0: Quick
	1: Lengthy

	----------------------------------

	Actions:

	Installation_Type
	0: Install_scrubbers
	1: New_cleaner_coal

	Strike_Intervention
	0: No
	1: Yes

	'''
	def __init__(self):

		#reward[(Coal_Worker_Strike, Strike_Resolution)][(Installation_Type, Strike_Intervention)]
		self.reward = {(1, 0): 
							{(1, 0): -3000000.0, 
							(0, 1): -3000000.0, 
							(1, 1): -4500000.0, 
							(0, 0): -3000000.0}, 
						(1, 1): 
							{(0, 0): -3000000.0, 
							(1, 0): -5000000.0, 
							(0, 1): -3000000.0, 
							(1, 1): -6500000.0}, 
						(0, 0): 
							{(1, 0): -1500000.0, 
							(0, 0): -3000000.0, 
							(0, 1): -3000000.0, 
							(1, 1): -1500000.0}, 
						(0, 1): 
							{(0, 0): -3000000.0, 
							(0, 1): -3000000.0, 
							(1, 0): -1500000.0, 
							(1, 1): -1500000.0}}
		self.tot_dec = 2
						
	def reset(self):
		
		self.Coal_Worker_Strike = np.nan
		self.Strike_Resolution = np.nan
		self.Installation_Type = np.nan
		self.Strike_Intervention = np.nan
		self.Additional_Cost = np.nan
			
		self.cur_dec = 0
		
			
		return self.state()
			
	def step(self, action):

		self.cur_dec += 1

		if self.cur_dec == 1:

			self.Installation_Type = action
			p = random.random()
			if p<0.6:
				self.Coal_Worker_Strike = 1
			else:
				self.Coal_Worker_Strike = 0

		elif self.cur_dec == 2:

			self.Strike_Intervention = action
			p = random.random()
			if self.Strike_Intervention == 1:
				if p<0.95:
					self.Strike_Resolution = 0
				else:
					self.Strike_Resolution = 1
			else:
				if p<0.70:
					self.Strike_Resolution = 0
				else:
					self.Strike_Resolution = 1
		
		if self.cur_dec == self.tot_dec:
			self.Additional_Cost =  self.reward[(self.Coal_Worker_Strike, self.Strike_Resolution)][(self.Installation_Type, self.Strike_Intervention)]
			return self.state(), self.Additional_Cost, True
		else:
			return self.state(), None, False
			
	def state(self):
		return [[self.Installation_Type, self.Coal_Worker_Strike, self.Strike_Intervention, self.Strike_Resolution, self.Additional_Cost]]

