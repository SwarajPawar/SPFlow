

import numpy as np
from collections import Counter

import pandas as pd
import random




class Test_Strep:

	'''
	---------------------------------------------------
	Variables and their encoded values:
	---------------------------------------------------



	State: 

	Test_Result		
	0: -ve
	1: +ve
	2: NA

	Streptococcal_Infection
	0: -ve
	1: +ve

	Rheumatic_Heart_Disease
	0: No
	1: Yes

	Die_from_Anaphylaxis
	0: No
	1: Yes

	Days_with_sore_throat
	3: Three
	4: Four

	-----------------------------------------------------

	Actions:

	Test_Decision
	0: No
	1: Yes

	Treatment_Decision	
	0: No
	1: Yes


	----------------------------------------------------


	'''

	def __init__(self):
		#self.reward[(Rheumatic_Heart_Disease, Die_from_Anaphylaxis, Days_with_sore_throat)]
		self.reward = {(0.0, 0.0, 4.0): 54.99, 
						(0.0, 0.0, 3.0): 54.996, 
						(1.0, 0.0, 3.0): 24.996, 
						(1.0, 0.0, 4.0): 24.995, 
						(0.0, 1.0, 4.0): 0.0,
						(0.0, 1.0, 3.0): 0.0, 
						(1.0, 1.0, 4.0): 0.0,
						(1.0, 1.0, 3.0): 0.0}
		self.tot_dec = 2

		#Done Indicator
		self.done = False
			

	#Initialize the variables to np.nan				
	def reset(self):
		
		self.Test_Decision = np.nan
		self.Streptococcal_Infection = np.nan
		self.Test_Result = np.nan
		self.Treatment_Decision = np.nan
		self.Rheumatic_Heart_Disease = np.nan
		self.Die_from_Anaphylaxis = np.nan
		self.Days_with_sore_throat = np.nan
		self.QALE = np.nan
			
		self.cur_dec = 0
		self.done = False
		
			
		return self.state()


	#Perform the action and get observed variables as per the CPTs		
	def step(self, action):
		
		self.cur_dec += 1

		if self.cur_dec == 1:

			self.Test_Decision = action
			p = random.random()
			if p<0.6:
				self.Streptococcal_Infection = 1
			else:
				self.Streptococcal_Infection = 0

			if self.Test_Decision == 1:
				p = random.random()
				if self.Streptococcal_Infection == 1:
					if p<0.9:
						self.Test_Result = 1
					else:
						self.Test_Result = 0
				else:
					if p<0.30:
						self.Test_Result = 1
					else:
						self.Test_Result = 0
			else:
				self.Test_Result = 2

		elif self.cur_dec == 2:

			self.Treatment_Decision = action

			self.Rheumatic_Heart_Disease = 0
			self.Die_from_Anaphylaxis = 0
			self.Days_with_sore_throat = 4

			if self.Treatment_Decision == 1:

				if self.Streptococcal_Infection == 1:

					self.Days_with_sore_throat = 3

					p = random.random()
					if p<0.0013:
						self.Rheumatic_Heart_Disease = 1
					else:
						self.Rheumatic_Heart_Disease = 0
				
				p = random.random()
				if p<0.001:
					self.Die_from_Anaphylaxis = 1

				
			elif self.Treatment_Decision == 0:
				if self.Streptococcal_Infection == 1:
					p = random.random()
					if p<0.0063:
						self.Rheumatic_Heart_Disease = 1

		#Return the reward if all actions are done
		if self.cur_dec == self.tot_dec:
			self.done = True
			
		if self.done:
			self.QALE =  self.reward[(self.Rheumatic_Heart_Disease, self.Die_from_Anaphylaxis, self.Days_with_sore_throat)]
			return self.state(), self.QALE, self.done
		else:
			return self.state(), None, self.done


	#Return the state as given by the partial order
	def state(self):
		
		if not self.done:
			return [[self.Test_Decision, np.nan, np.nan, self.Treatment_Decision, 
				np.nan, np.nan, np.nan, np.nan]]
		else:
			return [[self.Test_Decision, self.Streptococcal_Infection, self.Test_Result, self.Treatment_Decision, 
				self.Rheumatic_Heart_Disease, self.Die_from_Anaphylaxis, self.Days_with_sore_throat, self.QALE]]



