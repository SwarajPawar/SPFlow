

import numpy as np
from collections import Counter

import pandas as pd
import random

'''
Logic_board:
0: No
1: Yes

IO_board
0: No
1: Yes

Status
0: Failed
1: Operational

Outcome
0: L0_IO0
1: L0_IO1
2: L1_IO0



Decision:
0: Logic_board
1: IO_board

'''



'''
[('Logic_board', 'Yes', 'No', 'Failed', 'L0_IO1', 175), 
('Logic_board', 'Yes', 'Yes', 'Failed', 'L0_IO0', 300), 
('Logic_board', 'No', 'Yes', 'Failed', 'L1_IO0', 200), 
('Logic_board', 'No', 'No', 'Operational', 'L0_IO0', 300), 
('IO_board', 'No', 'No', 'Operational', 'L0_IO0', 300), 
('IO_board', 'No', 'Yes', 'Failed', 'L1_IO0', 125), 
('IO_board', 'Yes', 'Yes', 'Failed', 'L0_IO0', 300), 
('IO_board', 'Yes', 'No', 'Failed', 'L0_IO1', 225)]

Logic_board 	Counter({'Yes': 42126, 'No': 7874})      {'Yes': 0.84, 'No': 0.16}             
IO_board 		Counter({'No': 41549, 'Yes': 8451})		{'Yes': 0.17, 'No': 0.83}
'''
	
print(reward)


class Computer_Diagnostician:

	def __init__(self):
		self.rewards = {(1, 0, 0, 1): {0: 175, 1: 225}, 
						(1, 1, 0, 0): {1: 300, 0: 300}, 
						(0, 0, 0, 0): {1: 300, 0: 300}, 
						(0, 1, 0, 2): {1: 125, 0: 200}}


						
	def reset(self):
		
			
		p = random.random()
		if p<0.84:
			Logic_board = 1
		else:
			Logic_board = 0
			
		p = random.random()
		if p<0.17:
			IO_board = 1
		else:
			IO_board = 0
			
		if Logic_board == 1 and IO_board == 1:
			Status = 0
			Outcome = 0
		elif Logic_board == 1 and IO_board == 0:
			Status = 0
			Outcome = 1
		elif Logic_board == 0 and IO_board == 1:
			Status = 0
			Outcome = 2
		elif Logic_board == 0 and IO_board == 0:
			Status = 0
			Outcome = 0
			
		self.state = (Logic_board, IO_board, Status, Outcome)
			
	def step(self, action):
		
		return self.rewards[self.state][action]

