
import numpy as np
import math
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise

from spn.algorithms.splitting.Base import split_data_by_clusters, preproc
import logging

logger = logging.getLogger(__name__)
_rpy_initialized = False


import warnings
warnings.filterwarnings('ignore')




def split_rows_XMeans(local_data, seed=17):

	k =2
	while(True):

		kmeans = KMeans(n_clusters=k, random_state=seed)
		kmeans.fit(local_data)
		clusters = kmeans.labels_
		dim = np.size(local_data, axis=1)
		centers = kmeans.cluster_centers_
		p = dim + 1

		
		try:
			obic = np.zeros(k)

			for i in range(k):
				rn = np.size(np.where(clusters == i))
				var = np.sum((local_data[clusters == i] - centers[i])**2)/float(rn - 1)
				obic[i] = loglikelihood(rn, rn, var, dim, 1) - p/2.0*math.log(rn)

			sk = 2 #The number of subclusters
			nbic = np.zeros(k)
			addk = 0

			for i in range(k):
				ci = local_data[clusters == i]
				r = np.size(np.where(clusters == i))

				kmeans = KMeans(n_clusters=sk).fit(ci)
				ci_labels = kmeans.labels_
				sm = kmeans.cluster_centers_

				for l in range(sk):
					rn = np.size(np.where(ci_labels == l))
					var = np.sum((ci[ci_labels == l] - sm[l])**2)/float(rn - sk)
					nbic[i] += loglikelihood(r, rn, var, dim, sk)

				p = sk * (dim + 1)
				nbic[i] -= p/2.0*math.log(r)

				if obic[i] < nbic[i]:
					addk += 1
			if addk == 0:
				return k

			k = k + addk 
		except:
			return k





def loglikelihood(r, rn, var, m, k):
	l1 = - rn / 2.0 * math.log(2 * math.pi)
	l2 = - rn * m / 2.0 * math.log(var)
	l3 = - (rn - k) / 2.0
	l4 = rn * math.log(rn)
	l5 = - rn * math.log(r)
	return l1 + l2 + l3 + l4 + l5

import pandas as pd


datasets = ["nltcs","msnbc", "plants", "kdd", "baudio", "jester", "bnetflix"]


for dataset in datasets:
	
	print(f"\n\n\n{dataset}\n\n\n")
	
			
	df = pd.read_csv(f"spn/data/binary/{dataset}.ts.data", sep=',')
	data = df.values
	print(data.shape)
	k = split_rows_XMeans(data)
	print(k)
