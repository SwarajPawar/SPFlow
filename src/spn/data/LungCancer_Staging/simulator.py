

import numpy as np
from collections import Counter

import pandas as pd
import random


class LungCancer_Staging:

	'''
	---------------------------------------------------
	Variables and their encoded values:
	---------------------------------------------------


	State: 

	CTResult
	0: -ve
	1: +ve
	2: NA

	Mediastinal_Metastases
	0: No
	1: Yes

	Mediastinoscopy_Result
	0: -ve
	1: +ve
	2: NA

	Treatment_Death
	0: No
	1: Yes

	Mediastinoscopy_death
	0: No
	1: Yes

	-------------------------------------------------

	Actions:

	CT
	0: No
	1: Yes

	Treatment	
	0: Thoracotomy
	1: Radiation_therapy

	Mediastinoscopy
	0: No
	1: Yes

	---------------------------------------------------

	'''

	def __init__(self):
		#reward[med_met][Treatment]
		self.reward = {0.0:
							{0.0: 4.45, 
							1.0: 2.64},
						1.0:
							{0.0: 1.80, 
							1.0: 1.80}
						}
		self.tot_dec = 3

		#Done Indicator
		self.done = False



	#Initialize the variables to np.nan							
	def reset(self):
		
		self.CT = np.nan
		self.Mediastinal_Metastases = np.nan
		self.CTResult = np.nan
		self.Mediastinoscopy = np.nan
		self.Mediastinoscopy_Result = np.nan
		self.Mediastinoscopy_death = np.nan
		self.Treatment = np.nan
		self.Treatment_Death = np.nan
		self.Life_expectancy = np.nan
			
		self.cur_dec = 0
		self.done = False
		
			
		return self.state()
		
			
	#Perform the action and get observed variables as per the CPTs			
	def step(self, action):
		
		self.cur_dec += 1

		if self.cur_dec == 1:

			self.CT = action
			p = random.random()
			if p<0.46:
				self.Mediastinal_Metastases = 1
			else:
				self.Mediastinal_Metastases = 0

			if self.CT == 1:
				p = random.random()
				if self.Mediastinal_Metastases == 1:
					if p<0.82:
						self.CTResult = 1
					else:
						self.CTResult = 0
				else:
					if p<0.19:
						self.CTResult = 1
					else:
						self.CTResult = 0
			else:
				self.CTResult = 2

		elif self.cur_dec == 2:

			self.Mediastinoscopy = action

			self.Mediastinoscopy_Result = 2
			self.Mediastinoscopy_death = 0

			if self.Mediastinoscopy == 1:
				p = random.random()
				if self.Mediastinal_Metastases == 1:
					if p<0.82:
						self.Mediastinoscopy_Result = 1
					else:
						self.Mediastinoscopy_Result = 0
				else:
					if p<0.005:
						self.Mediastinoscopy_Result = 1
					else:
						self.Mediastinoscopy_Result = 0

				p = random.random()
				if p<0.005:
					self.Mediastinoscopy_death = 1
				
		elif self.cur_dec == 3:

			self.Treatment = action

			self.Treatment_Death = 0

			if self.Treatment == 0:
				p = random.random()
				if p<0.037:
					self.Treatment_Death = 1
			else:
				p = random.random()
				if p<0.002:
					self.Treatment_Death = 1

		#Return the reward if all actions are done
		if self.cur_dec == self.tot_dec:
			self.done = True
			
		if self.done:
			if self.Treatment_Death == 1 or self.Mediastinoscopy_death ==1:
				self.Life_expectancy =  0
			else:
				self.Life_expectancy =  self.reward[self.Mediastinal_Metastases][self.Treatment]
			return self.state(), self.Life_expectancy, self.done
		else:
			return self.state(), None, self.done


	#Return the state as given by the partial order
	def state(self):

		if not self.done:
			return [[self.CT, np.nan, np.nan, self.Mediastinoscopy, np.nan, np.nan, self.Treatment, np.nan, np.nan]]
		else:
			return [[self.CT, self.Mediastinal_Metastases, self.CTResult, self.Mediastinoscopy, 
				self.Mediastinoscopy_Result, self.Mediastinoscopy_death, self.Treatment, self.Treatment_Death, self.Life_expectancy]]