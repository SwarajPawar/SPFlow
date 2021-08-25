
import numpy as np

'''
	---------------------------------------------------
	Variables and their encoded values:
	---------------------------------------------------



	State:

	Elevator_at_Floor
	0, 1, 2

	Person_Waiting: 
	0: No
	1: Yes

	Person_in_Elevator_Going_Up: 
	0: No
	1: Yes

	Elevator_Direction: 
	0: Down
	1: Up

	Elevator_Closed:
	0: No
	1: Yes

	

	-----------------------------------------------------

	Actions:

	1: Open_Door_Going_Up
	2: Open_Door_Going_Down
	3: Move_Cur_Dir
	4: Close_Door


	----------------------------------------------------


'''

def convert_state_variables_Elevators(state):

	Elevator_at_Floor = list(state[0:3]).index(1)
	Person_Waiting = state[6]
	Person_in_Elevator_Going_Up = state[5]
	Elevator_Direction = state[4]
	Elevator_Closed = state[3]

	return [Elevator_at_Floor, Person_Waiting, Person_in_Elevator_Going_Up, Elevator_Direction, Elevator_Closed]

class Elevators:

	def __init__(self):
		self.decisions = 6
		self.info_set_size = 5


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