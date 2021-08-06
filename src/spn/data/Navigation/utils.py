
import numpy as np

'''
	---------------------------------------------------
	Variables and their encoded values:
	---------------------------------------------------



	State:

	Robot Locations
	| 1 | 2 |
	| 3 | 4 |
	| 5 | 6 |
	0: Robot Disappeared



	-----------------------------------------------------

	Actions:

	1: West
	2: North
	3: South
	4: East


	----------------------------------------------------


'''

def convert_state_variables_Navigation(state):


	Robot_at_1 = state[0]
	Robot_at_2 = state[2]
	Robot_at_3 = state[4]
	Robot_at_4 = state[5]
	Robot_at_5 = state[1]
	Robot_at_6 = state[3]

	Robot_Locations = [Robot_at_1, Robot_at_2, Robot_at_3, Robot_at_4, Robot_at_5, Robot_at_6]
	return Robot_Locations
	'''
	if 1 not in Robot_Locations:
		Robot_Location = 0
	else:
		Robot_Location = Robot_Locations.index(1)

	return [Robot_Location]
	'''


class Navigation:

	def __init__(self):
		self.decisions = 5
		self.info_set_size = 6


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