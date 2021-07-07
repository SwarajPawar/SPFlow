

import numpy as np
import collections

import pandas as pd
import random
import csv
import pickle

import matplotlib.pyplot as plt

from spn.data.simulator import get_env
import multiprocessing
from spn.data.metaData import *

a = [1,2,3]
b=tuple(a)
b += tuple((4))
for x in b:
	print(x)