
import numpy as np

'''
	---------------------------------------------------
	Variables and their encoded values:
	---------------------------------------------------



	State:

	HintDelayVarS0
	HintDelayVarS1
	HintedRightS0
	HintedRightS1
	ProficiencyMedS0
	ProficiencyMedS1
	UpdateTurnS0
	UpdateTurnS1
	AnsweredRightS0
	AnsweredRightS1
	ProficiencyHighS0
	ProficiencyHighS1
	
	


	-----------------------------------------------------

	Actions:

	1: GiveHintS1
	2: GiveHintS0
	3: AskProbS1
	4: AskProbS0

	
	


	----------------------------------------------------


'''

def convert_state_variables_SkillTeaching(state):


	
	return list(state)
	


class SkillTeaching:

	def __init__(self):
		self.decisions = 5
		self.info_set_size = 12


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