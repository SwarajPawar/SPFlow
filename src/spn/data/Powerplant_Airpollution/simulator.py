

import numpy as np
from collections import Counter

import pandas as pd
import random

	
data = pd.read_csv("Powerplant_Airpollution.tsv", sep="\t")

data = data.values[:,1:]

print(data.shape)
data = [tuple(x) for x in data]



unique = []

for x in data:
	if x not in unique:
		unique.append(x)
		
#freq1 = collections.Counter(np.array(data)[:,0])
#freq2 = collections.Counter(np.array(data)[:,1])
#freq3 = collections.Counter(np.array(data)[:,2])

#reward = {(x[0], x[1]):x[2] for x in unique}
#print(unique)
#print(freq1)
#print(freq2)
#print(freq3)

reward = dict()
for x in unique:
	if x[:3] not in reward:
		reward[x[:3]] = {x[3] : x[4]}
	else:
		reward[x[:3]][x[3]] = x[4]


freq = {('New_cleaner_coal', 'Yes', 'Quick'): 2522, ('Install_scrubbers', 'Yes', 'Quick'): 2468, ('Install_scrubbers', 'No', 'Quick'): 1683, ('New_cleaner_coal', 'No', 'Quick'): 1619, ('Install_scrubbers', 'Yes', 'Lengthy'): 531, ('New_cleaner_coal', 'Yes', 'Lengthy'): 488, ('New_cleaner_coal', 'No', 'Lengthy'): 350, ('Install_scrubbers', 'No', 'Lengthy'): 339}

Counter({'Install_scrubbers': 0.5, 'New_cleaner_coal': 0.5})
Counter({'Yes': 0.6, 'No': 0.4})
Counter({'Quick': 0.83, 'Lengthy': 0.17})


	
print(reward)


class Powerplant_Airpollution:

	def __init__(self):
		self.rewards = {('New_cleaner_coal', 'Yes', 'Quick'): {'No': -3000000.0, 'Yes': -4500000.0}, 
						('Install_scrubbers', 'Yes', 'Quick'): {'Yes': -3000000.0, 'No': -3000000.0}, 
						('Install_scrubbers', 'Yes', 'Lengthy'): {'No': -3000000.0, 'Yes': -3000000.0}, 
						('New_cleaner_coal', 'No', 'Quick'): {'No': -1500000.0, 'Yes': -1500000.0}, 
						('Install_scrubbers', 'No', 'Lengthy'): {'No': -3000000.0, 'Yes': -3000000.0}, 
						('Install_scrubbers', 'No', 'Quick'): {'No': -3000000.0, 'Yes': -3000000.0}, 
						('New_cleaner_coal', 'Yes', 'Lengthy'): {'No': -5000000.0, 'Yes': -6500000.0}, 
						('New_cleaner_coal', 'No', 'Lengthy'): {'No': -1500000.0, 'Yes': -1500000.0}}

						
	def reset(self):
		
		
		p = random.random()
		if p<0.5:
			Installation_Type = 'Install_scrubbers'
		else:
			Installation_Type = 'New_cleaner_coal'
			
		p = random.random()
		if p<0.6:
			Coal_Worker_Strike = 'Yes'
		else:
			Coal_Worker_Strike = 'No'
			
		p = random.random()
		if p<0.83:
			Strike_Resolution = 'Quick'
		else:
			Strike_Resolution = 'Lengthy'
			
		self.state = (Installation_Type, Coal_Worker_Strike, Strike_Resolution)
			
	def step(self, action):
		
		return self.rewards[self.state][action]

