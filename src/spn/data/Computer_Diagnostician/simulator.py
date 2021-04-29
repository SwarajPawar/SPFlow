

import numpy as np
from collections import Counter

import pandas as pd
import random


class Computer_Diagnostician:

	'''
	Logic_board_fail:
	0: No
	1: Yes

	IO_board_fail
	0: No
	1: Yes

	System_State
	0: Failed
	1: Operational

	self.Rework_Outcome
	0: L0_IO0
	1: L0_IO1
	2: L1_IO0



	Decision:
	0: Logic_board
	1: IO_board

	'''

	def __init__(self):

		#reward[Rework_Outcome][Rework_Decision]
		self.reward = {1: {0: 175, 1: 225}, 
						0: {1: 300, 0: 300},  
						2: {1: 125, 0: 200}}

		self.tot_dec = 1
						
	def reset(self):
		
		self.Logic_board_fail = np.nan
		self.IO_board_fail = np.nan
		self.System_State = np.nan
		self.Rework_Decision = np.nan
		self.Rework_Outcome = np.nan
		self.Rework_Cost = np.nan
			
		p = random.random()
		if p<0.84:
			self.Logic_board_fail = 1
		else:
			self.Logic_board_fail = 0
			
		p = random.random()
		if p<0.17:
			self.IO_board_fail = 1
		else:
			self.IO_board_fail = 0

		self.System_State = 0
		if self.Logic_board_fail == 0 and self.IO_board_fail == 0:
			self.System_State = 1
			
		
		return self.state()
			
	def step(self, action):

		self.Rework_Decision = action

		if self.Logic_board_fail == self.IO_board_fail:
			self.Rework_Outcome = 0
		elif self.Logic_board_fail == 1 and self.IO_board_fail == 0:
			self.Rework_Outcome = 1
		elif self.Logic_board_fail == 0 and self.IO_board_fail == 1:
			self.Rework_Outcome = 2
			
		self.Rework_Cost  = self.reward[self.Rework_Outcome][self.Rework_Decision]
		return self.state(), self.Rework_Cost, True

	def state(self):
		return [[self.Logic_board_fail, self.IO_board_fail, self.System_State, self.Rework_Decision, self.Rework_Outcome, self.Rework_Cost]]

