
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
	'msnbc': {"ll" : -6.035125836103926, 'nodes' : 5385, 'runtime' : 479.7606608867645},
	'kdd': {"ll" : -2.3006294371736202, 'nodes' : 8628, 'runtime' : 1634.041729927063},
	'jester': {"ll" : -51.13606060451481, 'nodes' : 27978, 'runtime' : 1164.8972017765045},
	'baudio': {"ll" : -39.67453486013839, 'nodes' : 17773, 'runtime' : 656.3769657611847},
	'bnetflix': {"ll" : -56.373594191535936, 'nodes' : 6074, 'runtime' : 1721.3557472229004},
	}


def get_original_stats(dataset):
	return original_stats[dataset]

def get_max_stats(dataset):
	return max_stats[dataset]