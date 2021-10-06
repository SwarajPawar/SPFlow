

import numpy as np

from spn.algorithms.ASPN import AnytimeSPN

from spn.algorithms.Statistics import get_structure_stats_dict
from spn.io.Graphics import plot_spn
from spn.data.domain_stats import get_original_stats, get_max_stats

from sklearn.model_selection import KFold
import logging
import random
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


import warnings
warnings.filterwarnings('ignore')



import pandas as pd
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType
from spn.io.ProgressBar import printProgressBar
import matplotlib.pyplot as plt
from os import path as pth
import sys, os



datasets = ["nltcs", "msnbc", "kdd", "jester", "baudio", "bnetflix"]

path = "cross_new"

#kfolds = 3
kfold = KFold(n_splits=3, shuffle=True)

for dataset in datasets:
	
	original = get_original_stats(dataset)
	upper = get_max_stats(dataset)



	'''
	k_ll = [[-2.5848930257656937, -2.492554556683935, -2.4699460596592715, -2.485947204444816, -2.478932879021837, -2.474265345228765, -2.459165161433595, -2.4481351394359643, -2.43541391933136, -2.4111312837817147, -2.402656573758984, -2.398774946424389, -2.3861493598526744, -2.386206914956455, -2.3881578702775825, -2.3726035357613267, -2.367617436933473, -2.3712830545309758, -2.363853309609781, -2.3595616790398855, -2.355629159081191, -2.3558392883183963, -2.3582596432873584, -2.3581110258541043, -2.352200097706046, -2.3574221680315017, -2.355822022289816, -2.349111216105615, -2.3491550530385723, -2.35513557621518, -2.3508660576500775, -2.350795606642518, -2.3503419288834544, -2.3485568854234344, -2.3513247121262935, -2.3533420574162394, -2.349680545165221, -2.3490648637870732, -2.349290771678356],
	[-2.554950340402117, -2.4814715508484797, -2.484871858133907, -2.468856312447519, -2.433418041624354, -2.441989038430403, -2.422706280174627, -2.432437729763897, -2.409062402144095, -2.406549295862202, -2.3901837872956264, -2.3859066586681, -2.3778445220300553, -2.3704351705732885, -2.3646867561644567, -2.3486356198160205, -2.3432886400493964, -2.3404154886466033, -2.361234722799295, -2.3431282197897816, -2.342740169747881, -2.341912407915938, -2.3389889391459966, -2.339456785864164, -2.33794039988021, -2.3323815073084435, -2.3378105172631827, -2.3355386661752475, -2.33311100978509, -2.332855153996417, -2.3330038300005707, -2.3262157033481747, -2.326010084390764, -2.3283477049151613, -2.328429853358591, -2.325551308957424, -2.3366851526853343, -2.3323906172616238, -2.3360538217293105, -2.336289149991864, -2.3362712464824593],
	[-2.5023877462290356, -2.53465895971052, -2.5059610238507046, -2.4591788471809184, -2.4820452503423183, -2.4470689361629154, -2.428985952943707, -2.4098502034089258, -2.4103557485206637, -2.4033197088070186, -2.3957445178074015, -2.389637013443353, -2.387492668172755, -2.375000292484424, -2.3666923686820236, -2.3610814898605614, -2.3580600056290977, -2.353035237504294, -2.3581135033275578, -2.3546582116506594, -2.353660183863746, -2.3465461789250326, -2.3461840632977373, -2.3454756421080245, -2.3501632351239534, -2.352006581753227, -2.3520510896440308, -2.3517837116283853, -2.3391141658408117, -2.3392891419362445, -2.3404397033038524, -2.3406315634062307, -2.3442475754459347, -2.345030878739896, -2.3426915558258448, -2.337581739679989, -2.337837535160206, -2.337277342434206]
	]
	k_nodes = [ [257, 342, 606, 673, 910, 959, 1067, 1133, 1412, 1661, 1827, 1902, 1820, 1982, 2188, 2034, 2184, 2705, 2300, 2335, 2475, 2473, 3266, 3407, 3283, 3050, 3229, 3757, 3873, 3466, 3573, 3968, 4085, 4197, 4201, 4154, 4540, 4723, 4767],
	[234, 339, 582, 590, 666, 945, 931, 1294, 1354, 1466, 1477, 1588, 1757, 1992, 2065, 2292, 2434, 2838, 2338, 2191, 2408, 2551, 2994, 3041, 3136, 3369, 3231, 3284, 3663, 3792, 3702, 4120, 4285, 4099, 4272, 4231, 4578, 4509, 4921, 5011, 4931],
	[238, 377, 526, 684, 829, 921, 916, 1374, 1288, 1468, 1477, 1688, 1751, 1796, 1985, 2123, 2599, 2544, 2553, 2552, 2845, 2867, 3053, 3200, 3136, 3218, 3238, 3418, 3398, 3584, 3734, 3604, 3869, 3856, 4470, 4247, 4432, 4488]

	]
	k_runtime = [[121.5805070400238, 122.43239688873291, 134.70038414001465, 135.73180294036865, 155.81128215789795, 185.6040380001068, 213.05036902427673, 258.8229682445526, 323.19156670570374, 379.3542821407318, 487.2824721336365, 542.7195255756378, 616.6663084030151, 724.7681114673615, 797.8456647396088, 1020.3750340938568, 1100.7287738323212, 1137.0116913318634, 1055.2339890003204, 1147.8669562339783, 1081.1817219257355, 1171.885666847229, 1333.7609813213348, 1204.65704703331, 1208.5971610546112, 1306.0788502693176, 1206.8925840854645, 1147.445966243744, 1115.2256717681885, 1172.050580739975, 1099.7792310714722, 1234.6142630577087, 1241.40917634964, 1198.9132840633392, 1189.912264585495, 1206.1336414813995, 1231.513226032257, 1176.1796803474426, 1144.1300542354584],
	[113.95119571685791, 119.49482369422913, 111.92976975440979, 118.0285632610321, 149.35787916183472, 184.2187945842743, 241.68556475639343, 261.8552792072296, 312.4952371120453, 369.24441957473755, 469.0537791252136, 523.0613067150116, 599.1466767787933, 633.2968108654022, 723.9173016548157, 852.6062791347504, 892.1539921760559, 1038.8310170173645, 1048.5279259681702, 1080.4417779445648, 1045.3835144042969, 1141.5853161811829, 1133.76619887352, 1192.3307378292084, 1240.9967126846313, 1248.4256796836853, 1152.933250427246, 1170.9861788749695, 1199.543399810791, 1291.223219871521, 1125.5882198810577, 1250.0956337451935, 1095.520387172699, 1272.5480768680573, 1195.4223618507385, 1154.166424036026, 1120.2062284946442, 1166.699743270874, 1108.0210485458374, 1080.5316414833069, 1104.5090658664703],
	[112.98651647567749, 103.68249249458313, 111.02359676361084, 136.2192211151123, 135.02109098434448, 186.35165929794312, 241.48774600028992, 274.47555589675903, 273.84203481674194, 389.3502519130707, 432.76310110092163, 588.3325757980347, 642.5678496360779, 711.0248277187347, 759.9419047832489, 868.635555267334, 1059.228303194046, 1127.983291387558, 1094.7519490718842, 1335.9156634807587, 1192.9061348438263, 1289.3157124519348, 1185.5494315624237, 1257.6282477378845, 1227.283230304718, 1192.5519652366638, 1224.667716741562, 1080.08406996727, 1186.9079375267029, 1195.1580593585968, 1094.1340825557709, 1110.560153245926, 1149.0786831378937, 1124.2151184082031, 1200.630603313446, 1012.2332727909088, 1198.6732287406921, 1061.4287128448486]

	]
	'''

	k_ll = list()
	k_nodes = list()
	k_runtime = list()

	for i in range(3):

		f = open(f"{path}/{dataset}/{i+1}/stats.txt", "r")
		f.readline()

		n = f.readline()
		n = n[n.index("[")+1:-2]
		n = n.split(", ")
		n = [float(x) for x in n]
		k_nodes.append(n)

		
		ll = f.readline()
		ll = ll[ll.index("[")+1:-2]
		ll = ll.split(", ")
		ll = [float(x) for x in ll]
		k_ll.append(ll)

		

		r = f.readline()
		r = r[r.index("[")+1:-2]
		r = r.split(", ")
		r = [float(x) for x in r]
		k_runtime.append(r)


		plt.plot(range(1,len(k_ll[i])+1), k_ll[i], marker="o")
		plt.title(f"{dataset} Log Likelihood")
		plt.xlabel("Iteration")
		plt.ylabel("Log Likelihood")
		plt.savefig(f"{path}/{dataset}/{i+1}/ll.png", dpi=150)
		plt.close()

		plt.plot(range(1,len(k_nodes[i])+1), k_nodes[i], marker="o")
		plt.title(f"{dataset} # Nodes")
		plt.xlabel("Iteration")
		plt.ylabel("Log Likelihood")
		plt.savefig(f"{path}/{dataset}/{i+1}/nodes.png", dpi=150)
		plt.close()

		plt.plot(range(1,len(k_runtime[i])+1), k_runtime[i], marker="o")
		plt.title(f"{dataset} Run Time (in seconds)")
		plt.xlabel("Iteration")
		plt.ylabel("Log Likelihood")
		plt.savefig(f"{path}/{dataset}/{i+1}/runtime.png", dpi=150)
		plt.close()
	

	
	plt.close()
	colors = ["red", "blue", "green"]

	maxlen = max([len(k_ll[i]) for i in range(len(k_ll))])
	total_ll = np.zeros(min([len(k_ll[i]) for i in range(len(k_ll))]))
	upperll = [upper["ll"]] * maxlen
	plt.plot(range(1, maxlen+1), upperll, linestyle="dashed", color ="darkred", linewidth=3, label="Upper Limit")
	originalll = [original["ll"]] * maxlen
	plt.plot(range(1, maxlen+1), originalll, linestyle="dotted", color ="purple", linewidth=3, label="LearnSPN")
	for i in range(len(k_ll)):
		plt.plot(range(1,len(k_ll[i])+1), k_ll[i], marker=f"{i+1}", color =colors[i], label=(i+1))
		total_ll += np.array(k_ll[i][:len(total_ll)])
	avg_ll = total_ll/len(k_ll)
	plt.plot(range(1,len(avg_ll)+1), avg_ll, marker="o", color ="black", linewidth=3, label="Mean")
	plt.title(f"{dataset} Log Likelihood")
	plt.legend()
	plt.xlabel("Iteration")
	plt.ylabel("Log Likelihood")
	plt.savefig(f"{path}/{dataset}/ll_{dataset}.png", dpi=150)
	plt.close()
	
	
	total_nodes = np.zeros(min([len(k_nodes[i]) for i in range(len(k_nodes))]))
	uppern = [upper["nodes"]] * maxlen
	plt.plot(range(1, maxlen+1), uppern, linestyle="dashed", color ="darkred", linewidth=3, label="Upper Limit")
	originaln = [original["nodes"]] * maxlen
	plt.plot(range(1, maxlen+1), originaln, linestyle="dotted", color ="purple", linewidth=3, label="LearnSPN")
	for i in range(len(k_nodes)):
		plt.plot(range(1,len(k_nodes[i])+1), k_nodes[i], marker=f"{i+1}", color =colors[i], label=(i+1))
		total_nodes += np.array(k_nodes[i][:len(total_nodes)])
	avg_nodes = total_nodes/len(k_nodes)
	plt.plot(range(1,len(avg_nodes)+1), avg_nodes, marker="o", color ="black", linewidth=3, label="Mean")
	plt.title(f"{dataset} Nodes")
	plt.legend()
	plt.xlabel("Iteration")
	plt.ylabel("# Nodes")
	plt.savefig(f"{path}/{dataset}/nodes_{dataset}.png", dpi=150)
	plt.close()


	total_time = np.zeros(min([len(k_runtime[i]) for i in range(len(k_runtime))]))
	uppertime = [upper["runtime"]] * maxlen
	plt.plot(range(1, maxlen+1), uppertime, linestyle="dashed", color ="darkred", linewidth=3, label="Upper Limit")
	originaltime = [original["runtime"]] * maxlen
	plt.plot(range(1, maxlen+1), originaltime, linestyle="dotted", color ="purple", linewidth=3, label="LearnSPN")
	for i in range(len(k_runtime)):
		plt.plot(range(1,len(k_runtime[i])+1), k_runtime[i], marker=f"{i+1}", color =colors[i], label=(i+1))
		total_time += np.array(k_runtime[i][:len(total_time)])
	avg_time = total_time/len(k_runtime)
	plt.plot(range(1,len(avg_time)+1), avg_time, marker="o", color ="black", linewidth=3, label="Mean")
	plt.title(f"{dataset} Run Time (in seconds)")
	plt.legend()
	plt.xlabel("Iteration")
	plt.ylabel("Run Time")
	plt.savefig(f"{path}/{dataset}/runtime_{dataset}.png", dpi=150)
	plt.close()
	
