
#Statistics for the LearnSPN algorithm
original_stats = {
	'nltcs': {"ll" : -6.387160274829616, 'nodes' : 114, 'runtime' : 15.676920652389526},
	'msnbc': {"ll" : -6.596837245245996, 'nodes' : 43, 'runtime' : 116.55100011825562},
	'kdd': {"ll" : -2.528963857512066, 'nodes' : 274, 'runtime' : 1096.580626964569},
	'jester': {"ll" : -57.09992785602156, 'nodes' : 255, 'runtime' : 226.89956641197205},
	'baudio': {"ll" : -43.40863103111932, 'nodes' : 322, 'runtime' : 288.9936385154724},
	'bnetflix': {"ll" : -61.26004536371138, 'nodes' : 350, 'runtime' : 301.1374011039734},
	'plants': {"ll" : -13.475238876574684, 'nodes' : 4783, 'runtime' : 2689.0314116477966},
	}

#Statistics for the LearnSPN with paremeters set to their max values
max_stats = {
	'nltcs': {"ll" : -5.983727733002763, 'nodes' : 2322, 'runtime' : 23.770670652389526},
	'msnbc': {"ll" : -6.035125836103926, 'nodes' : 5385, 'runtime' : 479.7606608867645},
	'kdd': {"ll" : -2.3006294371736202, 'nodes' : 8628, 'runtime' : 1634.041729927063},
	'jester': {"ll" : -51.13606060451481, 'nodes' : 27978, 'runtime' : 1164.8972017765045},
	'baudio': {"ll" : -39.07453486013839, 'nodes' : 17773, 'runtime' : 656.3769657611847},
	'bnetflix': {"ll" : -56.373594191535936, 'nodes' : 6074, 'runtime' : 1721.3557472229004},
	'plants': {"ll" : -12.494126455752165, 'nodes' : 24010, 'runtime' : 247.60532212257385},
	}


def get_original_stats(dataset):
	return original_stats[dataset]

def get_max_stats(dataset):
	return max_stats[dataset]