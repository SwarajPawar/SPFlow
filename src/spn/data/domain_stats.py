
#Statistics for the LearnSPMN algorithm
original_stats = {
	'Export_Textiles': {"ll" : -1.0890750655173789, "meu" : 1722313.8158882717, 'nodes' : 38, 'reward':1721301.8260000004, 'dev':3861.061525772288},
	'Test_Strep': {"ll" : -0.9130071749277912, "meu" : 54.9416526618876, 'nodes' : 130, 'reward':54.91352060400901, 'dev':0.013189836549851251},
	'LungCancer_Staging': {"ll" : -1.1489156814245234, "meu" : 3.138664586296027, 'nodes' : 312, 'reward':3.108005299999918, 'dev':0.011869627022775012},
	'HIV_Screening': {"ll" : -0.6276399171508842, "meu" : 42.582734183407034, 'nodes' : 112, 'reward':42.559788119992646, 'dev':0.06067708771159484},
	'Computer_Diagnostician': {"ll" : -0.9011245432112749, "meu" : -208.351, 'nodes' : 56, 'reward':-210.15520000000004, 'dev':0.3810022440878799},
	'Powerplant_Airpollution': {"ll" : -1.0796885930912947, "meu" : -2756263.244346315, 'nodes' : 38, 'reward':-2759870.4, 'dev':6825.630813338794},
	'Navigation': {"ll" : -0.25421657291373406, "meu" : -4.049331963001028, 'nodes' : 17426, 'edges' : 15274, 'layers' : 16, 'reward':-4.052, 'dev':0.022825424421026797, 'runtime':2044.684020280838},
	'Elevators': {"ll" : 0, "meu" : 0.5, 'nodes' : 38227, 'edges' : 32766, 'layers' : 14, 'reward':0.5, 'dev':0.0, 'runtime':2499.035037279129},
	'CrossingTraffic': {"ll" : -3.024408393022007, "meu" : -2.897592095165004, 'nodes' : 198840, 'edges' : 180126, 'layers' : 20, 'reward':-5.0, 'dev':0.0, 'runtime':6904.133388757706},
	'GameOfLife': {"ll" : -4.851702542481353, "meu" : 85.32740885779461, 'nodes' : 674834, 'edges' : 623912, 'layers' : 49, 'reward':11.294, 'dev':0.2812543332999511, 'runtime':18705.90127849579},
}

#Optimal MEU values from the IDs
optimal_meu = {
	'Export_Textiles' : 1721300,
	'Computer_Diagnostician': -210.13,
	'Powerplant_Airpollution': -2760000,
	'HIV_Screening': 42.5597,
	'Test_Strep': 54.9245,
	'LungCancer_Staging': 3.12453,
	'Navigation': -4.046906,
	'Elevators': 0.5,
	'CrossingTraffic': -4.2999,
	'CrossingTraffic1': -2.92106,
	'SkillTeaching': -7.180539,
	'GameOfLife': 11.567616 #10.808452,
}

#Rewards by simulating random policy
random_reward = {
	'Export_Textiles' : {'reward': 1300734.02, 'dev':7087.350616838437},
	'Computer_Diagnostician': {'reward': -226.666, 'dev':0.37205611135956335},
	'Powerplant_Airpollution': {'reward': -3032439.0, 'dev':7870.276615214995},
	'HIV_Screening': {'reward': 42.3740002199867, 'dev':0.07524234474837802},
	'Test_Strep': {'reward': 54.89614493400057, 'dev':0.012847272731391593},
	'LungCancer_Staging': {'reward': 2.672070640000026, 'dev':0.007416967451081523},
	'Navigation': {'reward': -4.972862, 'dev':0.00189885649800079},
	'CrossingTraffic': {'reward': -4.984152, 'dev':0.0007056174601014786},
	'CrossingTraffic1': {'reward': -4.779254, 'dev':0.005551430806557858},
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