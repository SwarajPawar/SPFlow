

import numpy as np
from collections import Counter

import pandas as pd
import random



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


'''
freq = {('New_cleaner_coal', 'Yes', 'Quick'): 2522, ('Install_scrubbers', 'Yes', 'Quick'): 2468, ('Install_scrubbers', 'No', 'Quick'): 1683, ('New_cleaner_coal', 'No', 'Quick'): 1619, ('Install_scrubbers', 'Yes', 'Lengthy'): 531, ('New_cleaner_coal', 'Yes', 'Lengthy'): 488, ('New_cleaner_coal', 'No', 'Lengthy'): 350, ('Install_scrubbers', 'No', 'Lengthy'): 339}

Counter({'Install_scrubbers': 0.5, 'New_cleaner_coal': 0.5})
Counter({'Yes': 0.6, 'No': 0.4})
Counter({'Quick': 0.83, 'Lengthy': 0.17})

'''



class Powerplant_Airpollution:

	def __init__(self):
		self.rewards = {(1, 0): 
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

						
	def reset(self):
		
		'''
		p = random.random()
		if p<0.5:
			Installation_Type = 'Install_scrubbers'
		else:
			Installation_Type = 'New_cleaner_coal'
		'''
			
		p = random.random()
		if p<0.6:
			Coal_Worker_Strike = 1
		else:
			Coal_Worker_Strike = 0
			
		p = random.random()
		if p<0.825:
			Strike_Resolution = 0
		else:
			Strike_Resolution = 1
			
		self.state = (Coal_Worker_Strike, Strike_Resolution)
			
	def step(self, action):
		
		return self.rewards[self.state][action]

