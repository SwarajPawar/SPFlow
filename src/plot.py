

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



datasets = ["nltcs"]

path = "cross_new"

#kfolds = 3
kfold = KFold(n_splits=3, shuffle=True)

for dataset in datasets:
	
	original = get_original_stats(dataset)
	upper = get_max_stats(dataset)


	k_ll = [[-6.8812825908727655, -6.541404496782005, -6.375147008958865, -6.361934062601527, -6.299387543205575, -6.242960326425666, -6.221992165847483, -6.186349127300236, -6.174665667602451, -6.163283180910645, -6.156571035206254, -6.160947143018406, -6.155808314113578, -6.151162868474467, -6.139309315630048, -6.127318818391569, -6.12975051272067, -6.124353015494685, -6.128280808449501, -6.113001448562086, -6.137360399549407, -6.1302666034611, -6.10995057564861, -6.1365351430671335, -6.129552763824225, -6.120601950765506, -6.122754340303972, -6.115059658215203, -6.113953116337573, -6.11615088672712],
			[-6.866535342501489, -6.51511079576476, -6.309569812210537, -6.261238835232845, -6.195807134658723, -6.155830008952676, -6.141948420902022, -6.115348632604652, -6.131839444651144, -6.130675903124943, -6.113706285100426, -6.093890917197712, -6.080197787559721, -6.077497464411195, -6.073926801815639, -6.0602555257591275, -6.042080156055706, -6.043227530186394, -6.055159272495919, -6.060522162198637, -6.061141756788678, -6.045044536957235, -6.053679428405561, -6.05194055683093, -6.046965938375607, -6.051529454039556, -6.050509475994903, -6.0581843062734375, -6.043990329091046, -6.044799933824482, -6.049846771214489, -6.050329319234075, -6.043792409382404, -6.044285800870778, -6.04246620029664],
			[-6.909223037673249, -6.558693522080958, -6.3663900024858755, -6.343994629610423, -6.2805867337231875, -6.219544364652479, -6.204801846598306, -6.173002942118286, -6.153603099563898, -6.143956573489844, -6.150175399323665, -6.126976650562471, -6.124884501233313, -6.117744613587375, -6.104903027866299, -6.108699420157624, -6.105443596078873, -6.100244810493881, -6.103187656184073, -6.102609591486124, -6.113233228128601, -6.0898958843908595, -6.110170744905389, -6.10076580789261, -6.079042705957797, -6.087291943905947, -6.0843151371524336, -6.082812554776044, -6.08094905303343]
			]
	k_nodes = [[68, 86, 115, 143, 153, 179, 212, 237, 248, 272, 284, 304, 324, 337, 374, 366, 394, 395, 452, 485, 522, 533, 492, 527, 556, 565, 602, 602, 620, 669],
				[68, 89, 114, 134, 170, 179, 211, 236, 247, 272, 300, 296, 315, 344, 377, 384, 408, 431, 422, 465, 467, 508, 543, 592, 577, 594, 586, 594, 658, 676, 688, 716, 732, 751, 764],
				[68, 81, 107, 134, 157, 183, 204, 241, 250, 277, 294, 305, 331, 329, 373, 393, 376, 389, 425, 481, 476, 505, 484, 516, 585, 622, 641, 657, 639]
				]
	k_runtime = [[4.734542369842529, 5.080339193344116, 7.890763521194458, 6.849035024642944, 9.421653747558594, 8.366650342941284, 10.150983572006226, 13.102485656738281, 14.591591596603394, 15.788973093032837, 14.607753038406372, 21.502428770065308, 16.84287714958191, 20.51817488670349, 22.442872285842896, 28.76084542274475, 27.325344800949097, 25.8677716255188, 30.233081579208374, 22.770222187042236, 23.44629955291748, 27.703943014144897, 24.675787448883057, 28.681859016418457, 30.521713972091675, 30.755229473114014, 30.447898149490356, 29.528112173080444, 27.218611478805542, 30.268990516662598],
				[4.384411811828613, 5.819538593292236, 7.40516996383667, 6.60754656791687, 9.497897386550903, 10.508217096328735, 11.126893520355225, 15.652820587158203, 21.388211011886597, 18.458417415618896, 19.734272956848145, 18.503678798675537, 17.462607860565186, 23.51068902015686, 17.172134160995483, 18.77042531967163, 22.185921669006348, 29.31935954093933, 31.57431674003601, 28.929622888565063, 28.900532007217407, 27.88497281074524, 32.47535252571106, 31.226478338241577, 34.04118013381958, 27.73843479156494, 31.773730039596558, 28.081456422805786, 38.19736051559448, 27.089319229125977, 31.55621838569641, 25.837501525878906, 28.80166459083557, 29.666117668151855, 29.813000440597534],
				[2.422173023223877, 4.908709287643433, 5.8110339641571045, 7.577314615249634, 8.956431865692139, 10.350857019424438, 16.368961572647095, 12.299113273620605, 18.564236640930176, 18.635128259658813, 18.443568229675293, 19.276851177215576, 19.797982692718506, 19.188735485076904, 25.480242490768433, 27.714974641799927, 26.671382188796997, 24.682653427124023, 24.325968742370605, 30.658201694488525, 35.23176026344299, 29.73987579345703, 27.009887218475342, 31.57456398010254, 33.83278822898865, 33.88709259033203, 33.04565095901489, 34.99703001976013, 29.780927419662476]
				]
	

	
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
	plt.savefig(f"{path}/{dataset}/ll.png", dpi=150)
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
	plt.savefig(f"{path}/{dataset}/nodes.png", dpi=150)
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
	plt.savefig(f"{path}/{dataset}/runtime.png", dpi=150)
	plt.close()
	
