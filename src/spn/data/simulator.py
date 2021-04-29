

import numpy as np
import collections

import pandas as pd
import random

from spn.data.simulator import Export_Textiles
from spn.data.simulator import Computer_Diagnostician
from spn.data.simulator import Powerplant_Airpollution
from spn.data.simulator import HIV_Screening
from spn.data.simulator import Test_Strep
from spn.data.simulator import LungCancer_Staging


def get_env(dataset):
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


