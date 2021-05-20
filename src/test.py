

import numpy as np
import collections

import pandas as pd
import random
import csv


import matplotlib.pyplot as plt

ll = [1000, 1010, 989]
dev = [10, 17, 12]
plt.errorbar(np.arange(len(ll)), ll, yerr=dev, marker="o", label="Anytime")
plt.axis(ymin=0)
plt.show()