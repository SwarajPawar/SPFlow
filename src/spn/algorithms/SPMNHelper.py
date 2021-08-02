"""
Created on March 28, 2019
@author: Hari Teja Tatavarti

"""

import numpy as np
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType
from spn.structure.leaves.parametric.Parametric import Categorical
from spn.algorithms.splitting.Clustering import loglikelihood
from spn.algorithms.splitting.Base import preproc, split_data_by_clusters
from sklearn.cluster import KMeans
import logging
import math


# below functions are used by learn_spmn_structure



def get_ds_context(data, scope, params):
	"""
	:param data: numpy array of data for Context object
	:param scope: scope of data
	:param params: params of SPMN
	:return: Context object of SPFlow
	"""

	num_of_variables = data.shape[1]
	scope_var = np.array(params.feature_names)[scope].tolist()
	ds_context = Context(
			meta_types=[params.meta_types[i] for i in scope],
			scope=scope,
			feature_names=scope_var
		)
	ds_context.add_domains(data)
# =======
#     # if parametric, all variables are type -- categorical
#     if params.util_to_bin:
#         context = [Categorical] * num_of_variables
#         ds_context = Context(parametric_types=context, scope=scope,
	#         feature_names=scope_var).add_domains(data)
#
#     # if mixed, utility is meta type -- UTILITY
#     else:
#
#         context = [MetaType.DISCRETE] * num_of_variables
#
#         utility_indices = [utility_index for utility_index,
	#         var in enumerate(scope_var)
#                            for utility_var in params.utility_nodes
#                            if utility_var == var]
#
#         # update context for utility variables with MetaType.UTILITY
#         if len(utility_indices) > 0:
#             for utility_index in utility_indices:
#                 context[utility_index] = MetaType.UTILITY
#
#         logging.debug(f'context is {context}')
#         scope = scope
#         ds_context = Context(meta_types=context, scope=scope,
	#         feature_names=scope_var).add_domains(data)
#
# >>>>>>> rspmn
	return ds_context


def cluster(data, dec_vals):
	"""
	:param data: numpy array of data containing variable at 0th column on whose values cluster is needed
	:param dec_vals: values of variable at that 0th column
	:return: clusters of data (excluding the variable at 0th column) grouped together based on values of the variable
	"""

	logging.debug(f'in cluster function of SPMNHelper')
	clusters_on_remaining_columns = []
	for i in range(0, len(dec_vals)):
		clustered_data_for_dec_val = data[[data[:, 0] == dec_vals[i]]]
		# exclude the 0th column, which belongs to decision node
		clustered_data_on_remaining_columns = np.delete(clustered_data_for_dec_val, 0, 1)
		# logging.debug(f'clustered data on remaining columns is {clustered_data_on_remaining_columns}')

		clusters_on_remaining_columns.append(clustered_data_on_remaining_columns)

	logging.debug(f'{len(clusters_on_remaining_columns)} clusters formed on remaining columns based on decision values')
	return clusters_on_remaining_columns


def split_on_decision_node(data):
	"""
	:param data: numpy array of data with decision node at 0th column
	:return: clusters split on values of decision node
	"""

	logging.debug(f'in split_on_decision_node function of SPMNHelper')
	# logging.debug(f'data at decision node is {data}')
	dec_vals = np.unique(data[:, 0])   # since 0th column of current train data is decision node
	logging.debug(f'dec_vals are {dec_vals}')
	# cluster remaining data based on decision values
	clusters_on_remaining_columns = cluster(data, dec_vals)
	return clusters_on_remaining_columns, dec_vals


def column_slice_data_by_scope(data, data_scope, slice_scope):
	"""
	:param data:  numpy array of data, columns ordered in data_scope order
	:param data_scope: scope of variables of the given data
	:param slice_scope: scope of the variables whose data slice is required
	:return: numpy array of data that corresponds to the variables of the given scope
	"""

	# assumption, data columns are ordered in data_scope order
	logging.debug(f'in column_slice_data_by_scope function of SPMNHelper')
	logging.debug(f'given scope of slice {slice_scope}')
	column_indices_of_slice_scope = [ind for ind, scope in enumerate(data_scope) if scope in slice_scope]
	logging.debug(f'column_indices_of_slice_scope are {column_indices_of_slice_scope}')

	data = data[:, column_indices_of_slice_scope]

	return data

def anytime_cluster(data, dec_vals):
	"""
	:param data: numpy array of data containing variable at 0th column on whose values cluster is needed
	:param dec_vals: groups of values of variable at that 0th column
	:return: clusters of data (excluding the variable at 0th column) grouped together based on the groups of values of the variable
	"""

	logging.debug(f'in cluster function of SPMNHelper')
	clusters_on_remaining_columns = []

	for i in range(0, len(dec_vals)):

		clustered_data_for_dec_val = None
		# For each variable in a group dec_vals[i], assign the respective data to the same cluster 
		for dec_val in dec_vals[i]:

			if clustered_data_for_dec_val is None:
				clustered_data_for_dec_val = data[[data[:, 0] == dec_val]]
			else:
				clustered_data_for_dec_val = np.concatenate((clustered_data_for_dec_val, data[[data[:, 0] == dec_val]]))
		# exclude the 0th column, which belongs to decision node
		clustered_data_on_remaining_columns = np.delete(clustered_data_for_dec_val, 0, 1)
		# logging.debug(f'clustered data on remaining columns is {clustered_data_on_remaining_columns}')

		clusters_on_remaining_columns.append(clustered_data_on_remaining_columns)

	logging.debug(f'{len(clusters_on_remaining_columns)} clusters formed on remaining columns based on decision values')
	return clusters_on_remaining_columns

def anytime_split_on_decision_node(data, m=None) :
	"""
	:param data: numpy array of data with decision node at 0th column
	:m: number of clusters to be formed
	:return: clusters split on values of decision node
	"""

	if m is None:
		return split_on_decision_node(data)

	logging.debug(f'in split_on_decision_node function of SPMNHelper')
	# logging.debug(f'data at decision node is {data}')
	dec_vals = np.unique(data[:, 0])   # since 0th column of current train data is decision node'
	logging.debug(f'dec_vals are {dec_vals}')

	#Prepare m groups
	m = min(m, len(dec_vals))
	dec_vals1 = [[dec_vals[i]] for i in range(m)]

	#Assign the decision values to the groups in a circular fashion
	for i in range(len(dec_vals) - m):
		dec_vals1[(i)%m].append(dec_vals[m + i])
	
	# cluster remaining data based on decision values
	clusters_on_remaining_columns = anytime_cluster(data, dec_vals1)
	return clusters_on_remaining_columns, dec_vals1
	

def get_split_rows_KMeans(n_clusters=2, pre_proc=None, ohe=False, seed=17):

	def split_rows_KMeans(local_data, ds_context, scope):
		
		if local_data.shape[0] == 1:
			local_data = np.concatenate((local_data, local_data))

		data = preproc(local_data, ds_context, pre_proc, ohe)

		km_model = KMeans(n_clusters=n_clusters, random_state=seed)
		clusters = km_model.fit_predict(data)
		return split_data_by_clusters(local_data, clusters, scope, rows=True), km_model

	return split_rows_KMeans

def get_split_rows_XMeans(pre_proc=None, ohe=False, seed=17, limit=math.inf, returnk = False, n=100, k=2):
	def split_rows_XMeans(local_data, ds_context, scope, k=k):

		
		if local_data.shape[0] == 1:
			local_data = np.concatenate((local_data, local_data))
		
		data = preproc(local_data, ds_context, pre_proc, ohe)

		km_model = None
		#prevk = k
		for i in range(n):

			km_model = KMeans(n_clusters=k, random_state=seed)
			km_model.fit(local_data)
			clusters = km_model.labels_
			dim = np.size(local_data, axis=1)
			centers = km_model.cluster_centers_
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
				#prevk = k

				k = k + addk 
				if (k + addk) >= limit:
					k= limit

			except:
				return split_data_by_clusters(local_data, clusters, scope, rows=True), km_model
		return split_data_by_clusters(local_data, clusters, scope, rows=True), km_model
	return split_rows_XMeans

def get_row_indices_of_cluster(labels_array, cluster_num):
	return np.where(labels_array == cluster_num)[0]


def row_slice_data_by_indices(data, indices):
	return data[indices, :]
