

import numpy as np
import collections

import pandas as pd
import random

from spn.data.Export_Textiles.simulator import Export_Textiles
from spn.data.Computer_Diagnostician.simulator import Computer_Diagnostician
from spn.data.Powerplant_Airpollution.simulator import Powerplant_Airpollution
from spn.data.HIV_Screening.simulator import HIV_Screening
from spn.data.Test_Strep.simulator import Test_Strep
from spn.data.LungCancer_Staging.simulator import LungCancer_Staging
from spn.data.FrozenLake.simulator import FrozenLake


#Return the simulator for the given dataset name
def get_env(dataset, return_state=False):
	if dataset == "Export_Textiles":
		return Export_Textiles()
	elif dataset == "Computer_Diagnostician":
		return Computer_Diagnostician()
	elif dataset == "Powerplant_Airpollution":
		return Powerplant_Airpollution()
	elif dataset == "HIV_Screening":
		return HIV_Screening()
	elif dataset == "Test_Strep":
		return Test_Strep()
	elif dataset == "LungCancer_Staging":
		return LungCancer_Staging()
	elif dataset == "FrozenLake":
		return FrozenLake(return_state)
	else:
		return None


