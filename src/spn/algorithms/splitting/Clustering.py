"""
Created on March 25, 2018

@author: Alejandro Molina
"""
import numpy as np
import math
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise

from spn.algorithms.splitting.Base import split_data_by_clusters, preproc
import logging

logger = logging.getLogger(__name__)
_rpy_initialized = False


def init_rpy():
	global _rpy_initialized
	if _rpy_initialized:
		return
	_rpy_initialized = True

	from rpy2 import robjects
	from rpy2.robjects import numpy2ri
	import os

	path = os.path.dirname(__file__)
	with open(path + "/mixedClustering.R", "r") as rfile:
		code = "".join(rfile.readlines())
		robjects.r(code)

	numpy2ri.activate()


def get_split_rows_KMeans(n_clusters=2, pre_proc=None, ohe=False, seed=17):
	def split_rows_KMeans(local_data, ds_context, scope):
		data = preproc(local_data, ds_context, pre_proc, ohe)

		clusters = KMeans(n_clusters=n_clusters, random_state=seed).fit_predict(data)

		return split_data_by_clusters(local_data, clusters, scope, rows=True)

	return split_rows_KMeans


def get_split_rows_XMeans(pre_proc=None, ohe=False, seed=17, limit=math.inf, returnk = True, n=100, k=2):
	def split_rows_XMeans(local_data, ds_context, scope, k=k):

		if local_data.shape[0] == 1:
			local_data = np.concatenate((local_data, local_data))
			
		data = preproc(local_data, ds_context, pre_proc, ohe)

		
		

		prevk = k
		for i in range(n):

			kmeans = KMeans(n_clusters=k, random_state=seed)
			kmeans.fit(local_data)
			clusters = kmeans.labels_
			dim = np.size(local_data, axis=1)
			centers = kmeans.cluster_centers_
			p = dim + 1

			
			if k>=limit:
				break
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
					if k + addk >= limit:
						break
				if addk == 0:
					break
				prevk = k

				k = k + addk 
				if (k + addk) >= limit:
					k= limit

			except:
				if returnk:
					return prevk, split_data_by_clusters(local_data, clusters, scope, rows=True)
				else:
					return split_data_by_clusters(local_data, clusters, scope, rows=True)
		if returnk:
			return k, split_data_by_clusters(local_data, clusters, scope, rows=True)
		else:
			return split_data_by_clusters(local_data, clusters, scope, rows=True)
	return split_rows_XMeans


def get_split_rows_TSNE(n_clusters=2, pre_proc=None, ohe=False, seed=17, verbose=10, n_jobs=-1):
	# https://github.com/DmitryUlyanov/Multicore-TSNE
	from MulticoreTSNE import MulticoreTSNE as TSNE
	import os

	ncpus = n_jobs
	if n_jobs < 1:
		ncpus = max(os.cpu_count() - 1, 1)

	def split_rows_KMeans(local_data, ds_context, scope):
		data = preproc(local_data, ds_context, pre_proc, ohe)
		kmeans_data = TSNE(n_components=3, verbose=verbose, n_jobs=ncpus, random_state=seed).fit_transform(data)
		clusters = KMeans(n_clusters=n_clusters, random_state=seed).fit_predict(kmeans_data)

		return split_data_by_clusters(local_data, clusters, scope, rows=True)

	return split_rows_KMeans


def get_split_rows_DBScan(eps=2, min_samples=10, pre_proc=None, ohe=False):
	def split_rows_DBScan(local_data, ds_context, scope):
		data = preproc(local_data, ds_context, pre_proc, ohe)

		clusters = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(data)

		return split_data_by_clusters(local_data, clusters, scope, rows=True)

	return split_rows_DBScan


def get_split_rows_Gower(n_clusters=2, pre_proc=None, seed=17):
	from rpy2 import robjects

	init_rpy()

	def split_rows_Gower(local_data, ds_context, scope):
		data = preproc(local_data, ds_context, pre_proc, False)

		try:
			df = robjects.r["as.data.frame"](data)
			clusters = robjects.r["mixedclustering"](df, ds_context.distribution_family, n_clusters, seed)
			clusters = np.asarray(clusters)
		except Exception as e:
			np.savetxt("/tmp/errordata.txt", local_data)
			logger.info(e)
			raise e

		return split_data_by_clusters(local_data, clusters, scope, rows=True)

	return split_rows_Gower


def get_split_rows_GMM(n_clusters=2, pre_proc=None, ohe=False, seed=17, max_iter=100, n_init=2, covariance_type="full"):
	"""
	covariance_type can be one of 'spherical', 'diag', 'tied', 'full'
	"""

	def split_rows_GMM(local_data, ds_context, scope):
		data = preproc(local_data, ds_context, pre_proc, ohe)

		estimator = GaussianMixture(
			n_components=n_clusters,
			covariance_type=covariance_type,
			max_iter=max_iter,
			n_init=n_init,
			random_state=seed,
		)

		clusters = estimator.fit(data).predict(data)

		return split_data_by_clusters(local_data, clusters, scope, rows=True)

	return split_rows_GMM




def loglikelihood(r, rn, var, m, k):
	l1 = - rn / 2.0 * math.log(2 * math.pi)
	l2 = - rn * m / 2.0 * math.log(var)
	l3 = - (rn - k) / 2.0
	l4 = rn * math.log(rn)
	l5 = - rn * math.log(r)
	return l1 + l2 + l3 + l4 + l5
