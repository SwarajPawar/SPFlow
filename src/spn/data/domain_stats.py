
#Statistics for the LearnSPMN algorithm
original_stats = {
	'Export_Textiles': {"ll" : -1.0890750655173789, "meu" : 1722313.8158882717, 'nodes' : 38, 'reward':1721301.8260000004, 'dev':3861.061525772288},
	'Test_Strep': {"ll" : -0.9130071749277912, "meu" : 54.9416526618876, 'nodes' : 130, 'reward':54.91352060400901, 'dev':0.013189836549851251},
	'LungCancer_Staging': {"ll" : -1.1489156814245234, "meu" : 3.138664586296027, 'nodes' : 312, 'reward':3.108005299999918, 'dev':0.011869627022775012},
	'HIV_Screening': {"ll" : -0.6276399171508842, "meu" : 42.582734183407034, 'nodes' : 112, 'reward':42.559788119992646, 'dev':0.06067708771159484},
	'Computer_Diagnostician': {"ll" : -0.9011245432112749, "meu" : -208.351, 'nodes' : 56, 'reward':-210.15520000000004, 'dev':0.3810022440878799},
	'Powerplant_Airpollution': {"ll" : -1.0796885930912947, "meu" : -2756263.244346315, 'nodes' : 38, 'reward':-2759870.4, 'dev':6825.630813338794}
}

#Optimal MEU values from the IDs
optimal_meu = {
	'Export_Textiles' : 1721300,
	'Computer_Diagnostician': -210.13,
	'Powerplant_Airpollution': -2760000,
	'HIV_Screening': 42.5597,
	'Test_Strep': 54.9245,
	'LungCancer_Staging': 3.12453
}

#Rewards by simulating random policy
random_reward = {
	'Export_Textiles' : {'reward': 1300734.02, 'dev':7087.350616838437},
	'Computer_Diagnostician': {'reward': -226.666, 'dev':0.37205611135956335},
	'Powerplant_Airpollution': {'reward': -3032439.0, 'dev':7870.276615214995},
	'HIV_Screening': {'reward': 42.3740002199867, 'dev':0.07524234474837802},
	'Test_Strep': {'reward': 54.89614493400057, 'dev':0.012847272731391593},
	'LungCancer_Staging': {'reward': 2.672070640000026, 'dev':0.007416967451081523},
}

def get_original_stats(dataset):
	return original_stats[dataset]

def get_optimal_meu(dataset):
	return optimal_meu[dataset]

def get_random_policy_reward(dataset):
	return random_reward[dataset]