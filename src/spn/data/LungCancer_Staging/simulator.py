

import numpy as np
from collections import Counter

import pandas as pd
import random



'''

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

----------------------------------

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


'''
	



class Powerplant_Airpollution:

	def __init__(self):
		#reward[med_met][Treatment]
		self.rewards = {0.0:
							{0.0: 4.45, 
							1.0: 2.64},
						1.0:
							{0.0: 1.80, 
							1.0: 1.80}
						}

						
	def reset(self):
		
		CTResult = 2
		Mediastinoscopy_Result = 2
		Treatment_Death = 0
		Mediastinoscopy_death = 0

		p = random.random()
		if p<0.46:
			Mediastinal_Metastases = 1
		else:
			Mediastinal_Metastases = 0
			
			
		self.state = (CTResult, Mediastinal_Metastases, Mediastinoscopy_Result, Treatment_Death, Mediastinoscopy_death)
			
	def step(self, action):
		
		CT, Treatment, Mediastinoscopy = action

		if CT == 1:
			p = random.random()
			if state[1] == 1:
				if p<0.82:
					self.state[0] = 1
				else:
					self.state[0] = 0
			else:
				if p<0.19:
					self.state[0] = 1
				else:
					self.state[0] = 0

		if Mediastinoscopy == 1:
			p = random.random()
			if state[1] == 1:
				if p<0.82:
					self.state[0] = 1
				else:
					self.state[0] = 0
			else:
				if p<0.005:
					self.state[0] = 1
				else:
					self.state[0] = 0

			p = random.random()
			if p<0.005:
				self.state[4] = 1


		if Treatment == 0:
			p = random.random()
			if p<0.037:
				self.state[3] = 1
		else:
			p = random.random()
			if p<0.002:
				self.state[3] = 1

		Treatment_Death, Mediastinoscopy_death = self.state[3], self.state[4]
		if Treatment_Death == 1 or Mediastinoscopy_death ==1:
			return 0
		else:
			return self.rewards[self.state[1]][Treatment]

