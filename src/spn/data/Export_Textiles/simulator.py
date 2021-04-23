

import numpy as np
import collections

import pandas as pd
import random



{('Same', 'After_12_mos', 210000.0): 1332, ('Same', 'Now', 2726000.0): 1329, ('Same', 'After_6_mos', 2357000.0): 1316, ('Slightly_worse', 'Now', 1870000.0): 1052, ('Slightly_worse', 'After_6_mos', 1694500.0): 1022, ('Severely_bad', 'Now', -711000.0): 1019, ('Severely_bad', 'After_12_mos', 766000.0): 1016, ('Severely_bad', 'After_6_mos', 900500.0): 982, ('Slightly_worse', 'After_12_mos', 1425000.0): 932}


class ExportTextiles:

	def __init__(self):
		self.rewards = {'Severely_bad': 
							{'After_6_mos': 900500.0, 
							'Now': -711000.0, 
							'After_12_mos': 766000.0}, 
						'Same':
							{'Now': 2726000.0, 
							'After_12_mos': 210000.0, 
							'After_6_mos': 2357000.0},
						'Slightly_worse':
							{ 'After_6_mos': 1694500.0,
							'Now': 1870000.0, 
							'After_12_mos': 1425000.0}
						}
						
	def reset(self):
		p = random.random()
		if p<0.4:
			self.state = 'Same'
		elif p<0.7:
			self.state = 'Severely_bad'
		else:
			self.state = 'Slightly_worse'
			
	def step(self, action):
		
		return self.rewards[self.state][action]

