

import numpy as np
from collections import Counter

import pandas as pd
import random




class HIV_Screening:

	'''
	---------------------------------------------------
	Variables and their encoded values:
	---------------------------------------------------


	State: 

	HIV_Test_Result
	0: -ve
	1: +ve
	2: NA

	HIV_Status
	0: -ve
	1: +ve

	Compliance_Medical_Therapy
	0: No
	1: Yes

	Reduce_Risky_Behavior
	0: No
	1: Yes

	---------------------------------------------------

	Actions:

	Screen
	0: No
	1: Yes

	Treat_Counsel
	0: No
	1: Yes


	----------------------------------------------------

	'''
		
	def __init__(self):

		#reward[(HIV_Status, Compliance_Medical_Therapy, Reduce_Risky_Behavior)]
		self.reward = {(1,1,1): 4.82,
						(1,1,0): 4.69,
						(1,0,1): 4.31,
						(1,0,0): 4.24,
						(0,1,1): 44.59,
						(0,1,0): 44.59,
						(0,0,1): 44.19,
						(0,0,0): 44.19
						}
		self.tot_dec = 2
					
		#Done Indicator
		self.done = False


	#Initialize the variables to np.nan		
	def reset(self):
		
		self.Screen = np.nan
		self.HIV_Status = np.nan
		self.HIV_Test_Result = np.nan
		self.Treat_Counsel = np.nan
		self.Compliance_Medical_Therapy = np.nan
		self.Reduce_Risky_Behavior  = np.nan
		self.QALE = np.nan
			
		self.cur_dec = 0
		self.done = False
		

		
			
			
		return self.state()
	

	#Perform the action and get observed variables as per the CPTs				
	def step(self, action):
		
		self.cur_dec += 1

		if self.cur_dec == 1:

			self.Screen = action
			p = random.random()
			if p<0.05:
				self.HIV_Status = 1
			else:
				self.HIV_Status = 0

			if self.Screen == 1:
				p = random.random()
				if self.HIV_Status == 1:
					if p<0.995:
						self.HIV_Test_Result = 1
					else:
						self.HIV_Test_Result = 0
				else:
					if p<0.00006:
						self.HIV_Test_Result = 1
					else:
						self.HIV_Test_Result = 0
			else:
				self.HIV_Test_Result = 2

		elif self.cur_dec == 2:

			self.Treat_Counsel = action
			self.Compliance_Medical_Therapy, self.Reduce_Risky_Behavior = 0, 0
			if self.Treat_Counsel == 1:
				p = random.random()
				if p<0.9:
					self.Compliance_Medical_Therapy = 1
				p = random.random()
				if p<0.8:
					self.Reduce_Risky_Behavior = 1

		#Return the reward if all actions are done
		if self.cur_dec == self.tot_dec:
			self.done = True
			
		if self.done:
			self.QALE =  self.reward[(self.HIV_Status, self.Compliance_Medical_Therapy, self.Reduce_Risky_Behavior)]
			return self.state(), self.QALE, self.done
		else:
			return self.state(), None, self.done


	#Return the state as given by the partial order
	def state(self):
		if not self.done:
			return [[self.Screen, np.nan, np.nan, self.Treat_Counsel, np.nan, np.nan, np.nan]]
		else:
			return [[self.Screen, self.HIV_Status, self.HIV_Test_Result, self.Treat_Counsel,
				 self.Compliance_Medical_Therapy, self.Reduce_Risky_Behavior, self.QALE]]
