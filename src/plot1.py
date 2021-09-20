

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



datasets = ["msnbc"]

path = "cross_new1/msnbc/1"



ll = [-6.625441787462082, -6.59532671476254, -6.581436579852523, -6.5150524463171955, -6.46282141089309, -6.443083619793373, -6.432239334097712, -6.432718110779891, -6.421647202908297, -6.370522157411557, -6.380146520913458, -6.3615670549558185, -6.337823744137224, -6.3017037167801275, -6.289557130951925, -6.2955969161504015, -6.284010500028773, -6.275809645721455, -6.268039868941022, -6.264248695196058, -6.272118997104198, -6.245759680958601, -6.27543280307744, -6.27505807114179, -6.22010495779757, -6.221286303901232, -6.2131266854621865, -6.215790125981312, -6.2151161637227785, -6.208863899635914, -6.201995471393991, -6.170300532788632, -6.203663966840016, -6.200650489512748, -6.19902236932335, -6.195635711926318, -6.188951208164091, -6.191831488438949, -6.1848322138530865, -6.174407943881358, -6.165545639572256, -6.154941493034209, -6.146883447735869, -6.146735044138365, -6.146723875768395]





plt.close()
plt.plot(range(1,len(ll)+1), ll, marker="o", label="Anytime")
plt.title(f"{dataset} Log Likelihood")
plt.xlabel("Iteration")
plt.ylabel("Log Likelihood")
plt.legend()
plt.savefig(f"{path}/ll.png", dpi=100)
plt.close()
	
