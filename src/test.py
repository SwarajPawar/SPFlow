

import numpy as np
import collections

import pandas as pd
import random
import csv


import matplotlib.pyplot as plt

from spn.data.simulator import get_env
import multiprocessing

dataset = 'FrozenLake'
df = pd.read_csv(f"spn/data/{dataset}/{dataset}_new.tsv", sep='\t')

data = df.values
reward = data[:,-1]
from collections import Counter
print(Counter(reward))
