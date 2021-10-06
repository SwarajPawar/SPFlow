
#Statistics for the LearnSPN algorithm
original_stats = {
	'nltcs': {"ll" : -6.410009094638417, 'nodes' : 114, 'runtime' : 15.197888612747192},
	'msnbc': {"ll" : -6.596712414458973, 'nodes' : 43, 'runtime' : 104.47390723228455},
	'kdd': {"ll" : -2.2637023576924853, 'nodes' : 294, 'runtime' : 987.3730320930481},
	'jester': {"ll" : -56.95361505735743, 'nodes' : 255, 'runtime' : 181.81801533699036},
	'baudio': {"ll" : -43.25947745856322, 'nodes' : 333, 'runtime' : 280.24337911605835},
	'bnetflix': {"ll" : -61.116128202699464, 'nodes' : 342, 'runtime' : 237.63625717163086},
	}

#Statistics for the LearnSPN with paremeters set to their max values
max_stats = {
	'nltcs': {"ll" : -6.03977180079453, 'nodes' : 2280, 'runtime' : 21.817023992538452},
	'msnbc': {"ll" : -6.0454543442301185, 'nodes' : 5401, 'runtime' : 428.08805108070374},
	'kdd': {"ll" : -2.1597072441370195, 'nodes' : 4108, 'runtime' : 1487.8591768741608},
	'jester': {"ll" : -52.987940009436976, 'nodes' : 13728, 'runtime' : 915.3376817703247},
	'baudio': {"ll" : -39.981952364753546, 'nodes' : 13491, 'runtime' : 815.7641651630402},
	'bnetflix': {"ll" : -56.88984026246377, 'nodes' : 4718, 'runtime' : 1222.8357133865356},
	}


def get_original_stats(dataset):
	return original_stats[dataset]

def get_max_stats(dataset):
	return max_stats[dataset]