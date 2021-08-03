
import numpy as np

'''
	---------------------------------------------------
	Variables and their encoded values:
	---------------------------------------------------



	State:

	Cell Locations
	| 1 | 4 | 7 |
	| 2 | 5 | 8 |
	| 3 | 6 | 9 |
	
	0: Dead
	1: Alive



	-----------------------------------------------------

	Actions:

	1: Set 1
	2: Set 2
	3: Set 3
	4: Set 4
	5: Set 5
	6: Set 6
	7: Set 7
	8: Set 8
	9: Set 9


	----------------------------------------------------


'''

def convert_state_variables_GameOfLife(state):


	Cell_1 = state[0]
	Cell_2 = state[1]
	Cell_3 = state[2]
	Cell_4 = state[3]
	Cell_5 = state[4]
	Cell_6 = state[5]
	Cell_7 = state[6]
	Cell_8 = state[7]
	Cell_9 = state[8]

	cells = [Cell_1, Cell_2, Cell_3, Cell_4, Cell_5, Cell_6, Cell_7, Cell_8, Cell_9]
	return cells
	


class GameOfLife:

	def __init__(self):
		self.decisions = 3
		self.info_set_size = 9


	def reset(self):
		self.actions = [np.nan for i in range(self.decisions)]
		self.cur_action = 0
		return self.get_sequence()


	def next_complete_sequence(self, action):

		self.actions[self.cur_action] = action

		self.cur_action += 1

		return self.get_sequence()

	def get_sequence(self):
		sequence = list()
		for i in range(self.decisions):
			sequence += [np.nan]*self.info_set_size + [self.actions[i]]

		sequence += [np.nan]*(self.info_set_size + 1)

		return [sequence]