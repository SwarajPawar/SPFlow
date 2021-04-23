

import numpy as np
from collections import Counter

import pandas as pd
import random

	
data = pd.read_csv("Computer_Diagnostician.tsv", sep="\t")

data = data.values[:,1:]

print(data.shape)
data = [tuple(x) for x in data]



unique = []

for x in data:
	if x not in unique:
		unique.append(x)
		
#freq1 = Counter(np.array(data)[:,0])
freq2 = Counter(np.array(data)[:,1])
freq3 = Counter(np.array(data)[:,2])

#reward = {(x[0], x[1]):x[2] for x in unique}
#print(unique)
#print(freq1)
print(freq2)
print(freq3)


reward = dict()
for x in unique:
	if x[1:5] not in reward:
		reward[x[1:5]] = {x[0] : x[5]}
	else:
		reward[x[1:5]][x[0]] = x[5]

'''
[('Logic_board', 'Yes', 'No', 'Failed', 'L0_IO1', 175), 
('Logic_board', 'Yes', 'Yes', 'Failed', 'L0_IO0', 300), 
('Logic_board', 'No', 'Yes', 'Failed', 'L1_IO0', 200), 
('Logic_board', 'No', 'No', 'Operational', 'L0_IO0', 300), 
('IO_board', 'No', 'No', 'Operational', 'L0_IO0', 300), 
('IO_board', 'No', 'Yes', 'Failed', 'L1_IO0', 125), 
('IO_board', 'Yes', 'Yes', 'Failed', 'L0_IO0', 300), 
('IO_board', 'Yes', 'No', 'Failed', 'L0_IO1', 225)]

Logic_board 	Counter({'Yes': 42126, 'No': 7874})      {'Yes': 0.85, 'No': 0.15}             
IO_board 		Counter({'No': 41549, 'Yes': 8451})		{'Yes': 0.83, 'No': 0.17}
'''
	
print(reward)


class Computer_Diagnostician:

	def __init__(self):
		self.rewards = {('Yes', 'No', 'Failed', 'L0_IO1'): {'Logic_board': 175, 'IO_board': 225}, 
						('Yes', 'Yes', 'Failed', 'L0_IO0'): {'IO_board': 300, 'Logic_board': 300}, 
						('No', 'No', 'Operational', 'L0_IO0'): {'IO_board': 300, 'Logic_board': 300}, 
						('No', 'Yes', 'Failed', 'L1_IO0'): {'IO_board': 125, 'Logic_board': 200}}


						
	def reset(self):
		
			
		p = random.random()
		if p<0.6:
			Logic_board = 'Yes'
		else:
			Logic_board = 'No'
			
		p = random.random()
		if p<0.6:
			IO_board = 'Yes'
		else:
			IO_board = 'No'
			
		if Logic_board == 'Yes' and IO_board == 'Yes':
			Status = 'Failed'
			Outcome = 'L0_IO0'
		elif Logic_board == 'Yes' and IO_board == 'No':
			Status = 'Failed'
			Outcome = 'L0_IO1'
		elif Logic_board == 'No' and IO_board == 'Yes':
			Status = 'Failed'
			Outcome = 'L1_IO0'
		elif Logic_board == 'No' and IO_board == 'No':
			Status = 'Operational'
			Outcome = 'L0_IO0'
			
		self.state = (Logic_board, IO_board, Status, Outcome)
			
	def step(self, action):
		
		return self.rewards[self.state][action]

