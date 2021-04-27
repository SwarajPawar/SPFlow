

import numpy as np
from collections import Counter

import pandas as pd
import random



'''

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

----------------------------------

Actions:

Screen
0: No
1: Yes

Treat_Counsel
0: No
1: Yes




'''
	


class Powerplant_Airpollution:

	def __init__(self):
		self.rewards = {(1,1,1): 4.82,
						(1,1,0): 4.69,
						(1,0,1): 4.31,
						(1,0,0): 4.24,
						(0,1,1): 44.59,
						(0,1,0): 44.59,
						(0,0,1): 44.19,
						(0,0,0): 44.19
						}

						
	def reset(self):
		
		HIV_Test_Result = 2
		Compliance_Medical_Therapy = 0
		Reduce_Risky_Behavior = 0

		p = random.random()
		if p<0.05:
			HIV_Status = 1
		else:
			HIV_Status = 0
			
			
		self.state = (HIV_Test_Result, HIV_Status, Compliance_Medical_Therapy, Reduce_Risky_Behavior)
			
	def step(self, action):
		
		Screen, Treat_Counsel = action

		if Screen == 1:
			p = random.random()
			if state[1] == 1:
				if p<0.995:
					self.state[0] = 1
				else:
					self.state[0] = 0
			else:
				if p<0.00006:
					self.state[0] = 1
				else:
					self.state[0] = 0

		if Treat_Counsel == 1:
			p = random.random()
			if p<0.9:
				self.state[2] = 1
			p = random.random()
			if p<0.8:
				self.state[3] = 1

		return self.rewards[self.state[1:]]

