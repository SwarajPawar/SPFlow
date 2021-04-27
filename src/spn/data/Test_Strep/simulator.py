

import numpy as np
from collections import Counter

import pandas as pd
import random



'''

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

----------------------------------

Actions:

Test_Decision
0: No
1: Yes

Treatment_Decision	
0: No
1: Yes





'''
	
data = pd.read_csv("Test_Strep_enc.tsv", sep="\t")

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
	if (x[1], x[2], x[4], x[5], x[6]) not in reward:
		reward[(x[1], x[2], x[4], x[5], x[6])] = {(x[0], x[3]) : x[-1]}
	else:
		reward[(x[1], x[2], x[4], x[5], x[6])][(x[0], x[3])] = x[-1]

'''
freq = {('New_cleaner_coal', 'Yes', 'Quick'): 2522, ('Install_scrubbers', 'Yes', 'Quick'): 2468, ('Install_scrubbers', 'No', 'Quick'): 1683, ('New_cleaner_coal', 'No', 'Quick'): 1619, ('Install_scrubbers', 'Yes', 'Lengthy'): 531, ('New_cleaner_coal', 'Yes', 'Lengthy'): 488, ('New_cleaner_coal', 'No', 'Lengthy'): 350, ('Install_scrubbers', 'No', 'Lengthy'): 339}

Counter({'Install_scrubbers': 0.5, 'New_cleaner_coal': 0.5})
Counter({'Yes': 0.6, 'No': 0.4})
Counter({'Quick': 0.83, 'Lengthy': 0.17})

'''
	
print(reward)


class Powerplant_Airpollution:

	def __init__(self):
		self.rewards = {(0.0, 0.0, 4.0): 54.99, 
						(0.0, 0.0, 3.0): 54.996, 
						(1.0, 0.0, 3.0): 24.996, 
						(1.0, 0.0, 4.0): 24.995, 
						(0.0, 1.0, 4.0): 0.0,
						(0.0, 1.0, 3.0): 0.0, 
						(1.0, 1.0, 4.0): 0.0,
						(1.0, 1.0, 3.0): 0.0}

						
	def reset(self):
		
		Test_Result = 2
		Rheumatic_Heart_Disease = 0
		Die_from_Anaphylaxis = 0
		Days_with_sore_throat = 4

		p = random.random()
		if p<0.6:
			Streptococcal_Infection = 1
		else:
			Streptococcal_Infection = 0
			
			
		self.state = (Test_Result, Streptococcal_Infection, Rheumatic_Heart_Disease, Die_from_Anaphylaxis, Days_with_sore_throat)
			
	def step(self, action):
		
		Test_Decision, Treatment_Decision = action

		if Test_Decision == 1:
			p = random.random()
			if p<0.66:
				self.state[0] = 1
			else:
				self.state[0] = 0

		if Treatment_Decision == 1:
			p = random.random()
			if p<0.001:
				self.state[3] = 1

			p = random.random()
			if p<0.00078:
				self.state[2] = 1

			p = random.random()
			if p<0.6:
				self.state[4] = 3

		elif Treatment_Decision == 0:
			p = random.random()
			if p<0.0038:
				self.state[2] = 1

		return self.rewards[self.state[2:]]

