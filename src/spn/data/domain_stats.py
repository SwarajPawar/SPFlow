
#Statistics for the LearnSPMN algorithm
original_stats = {
	'Export_Textiles': {"ll" : -1.0890750655173789, "meu" : 1722313.8158882717, 'nodes' : 38, 'reward':1721301.8260000004, 'dev':3861.061525772288},
	'Test_Strep': {"ll" : -0.9130071749277912, "meu" : 54.9416526618876, 'nodes' : 130, 'reward':54.91352060400901, 'dev':0.013189836549851251},
	'LungCancer_Staging': {"ll" : -1.1489156814245234, "meu" : 3.138664586296027, 'nodes' : 312, 'reward':3.108005299999918, 'dev':0.011869627022775012},
	'HIV_Screening': {"ll" : -0.6276399171508842, "meu" : 42.582734183407034, 'nodes' : 112, 'reward':42.559788119992646, 'dev':0.06067708771159484},
	'Computer_Diagnostician': {"ll" : -0.9011245432112749, "meu" : -208.351, 'nodes' : 56, 'reward':-210.15520000000004, 'dev':0.3810022440878799},
	'Powerplant_Airpollution': {"ll" : -1.0796885930912947, "meu" : -2756263.244346315, 'nodes' : 38, 'reward':-2759870.4, 'dev':6825.630813338794},
	'Navigation': {"ll" : -0.25421657291373406, "meu" : -4.049331963001028, 'nodes' : 17426, 'edges' : 15274, 'layers' : 16, 'reward':-4.052, 'dev':0.022825424421026797, 'runtime':2044.684020280838},
	'Navigation1': {"ll" : -0.24199351938969704, "meu" : -3.120315581854044, 'nodes' : 17338, 'reward':-3.1046000000000005, 'dev':0.042266298631415615, 'runtime':3610.374293088913},
	'Elevators': {"ll" : 0, "meu" : 0.5, 'nodes' : 38227, 'edges' : 32766, 'layers' : 14, 'reward':0.5, 'dev':0.0, 'runtime':2499.035037279129},
}

#Optimal MEU values from the IDs
optimal_meu = {
	'Export_Textiles' : 1721300,
	'Computer_Diagnostician': -210.13,
	'Powerplant_Airpollution': -2760000,
	'HIV_Screening': 42.5597,
	'Test_Strep': 54.9245,
	'LungCancer_Staging': 3.12453,
	'Navigation1': -3.09726,
	'Navigation': -4.046906,
	'Elevators': 0.5
}

#Rewards by simulating random policy
random_reward = {
	'Export_Textiles' : {'reward': 1300734.02, 'dev':7087.350616838437},
	'Computer_Diagnostician': {'reward': -226.666, 'dev':0.37205611135956335},
	'Powerplant_Airpollution': {'reward': -3032439.0, 'dev':7870.276615214995},
	'HIV_Screening': {'reward': 42.3740002199867, 'dev':0.07524234474837802},
	'Test_Strep': {'reward': 54.89614493400057, 'dev':0.012847272731391593},
	'LungCancer_Staging': {'reward': 2.672070640000026, 'dev':0.007416967451081523},
	'Navigation1': {'reward': -4.941799999999999, 'dev':0.018664404624846833},
	'Navigation': {'reward': -4.972862, 'dev':0.00189885649800079},
	'CrossingTraffic': {'reward': -4.779254, 'dev':0.005551430806557858},
	'Elevators': {'reward': -6.129405499999999 , 'dev':0.006561652878657956},
	'GameOfLife': {'reward': 9.671382000000001 , 'dev':0.02171178887148651},
	'SysAdmin': {'reward': 12.110048999999998 , 'dev':0.02391732450547092},
}

def get_original_stats(dataset):
	return original_stats[dataset]

def get_optimal_meu(dataset):
	return optimal_meu[dataset]

def get_random_policy_reward(dataset):
	return random_reward[dataset]