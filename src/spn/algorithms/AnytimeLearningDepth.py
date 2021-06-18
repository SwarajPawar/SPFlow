"""
Created on March 20, 2018

@author: Alejandro Molina
"""
import logging
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)

try:
	from time import perf_counter
except:
	from time import time

	perf_counter = time

import numpy as np

from spn.algorithms.TransformStructure import Prune, Copy
from spn.algorithms.Validity import is_valid
from spn.algorithms.Statistics import get_structure_stats
from spn.structure.Base import Product, Sum, assign_ids
from spn.algorithms.splitting.Clustering import get_split_rows_XMeans
from spn.algorithms.splitting.RDC import get_split_cols_single_RDC
import multiprocessing
import os

parallel = True

if parallel:
	cpus = max(1, os.cpu_count() - 2)  # - int(os.getloadavg()[2])
else:
	cpus = 1
pool = multiprocessing.Pool(processes=cpus)


class Operation(Enum):
	CREATE_LEAF = 1
	SPLIT_COLUMNS = 2
	SPLIT_ROWS = 3
	NAIVE_FACTORIZATION = 4
	REMOVE_UNINFORMATIVE_FEATURES = 5
	CONDITIONING = 6


def get_next_operation(min_instances_slice=100, min_features_slice=1, multivariate_leaf=False):
	def next_operation(
		data,
		scope,
		create_leaf,
		no_clusters=False,
		no_independencies=False,
		is_first=False,
		cluster_first=True,
		cluster_univariate=False,
		param = None
	):

		minimalFeatures = len(scope) == min_features_slice
		minimalInstances = data.shape[0] <= min_instances_slice

		if minimalFeatures:
			if minimalInstances or no_clusters:
				return Operation.CREATE_LEAF, None
			else:
				if cluster_univariate:
					if param is None:
						return Operation.SPLIT_ROWS, None
					else:
						return Operation.SPLIT_ROWS, param
				else:
					return Operation.CREATE_LEAF, None

		uninformative_features_idx = np.var(data[:, 0 : len(scope)], 0) == 0
		ncols_zero_variance = np.sum(uninformative_features_idx)
		if ncols_zero_variance > 0:
			if ncols_zero_variance == data.shape[1]:
				if multivariate_leaf:
					return Operation.CREATE_LEAF, None
				else:
					return Operation.NAIVE_FACTORIZATION, None
			else:
				return (
					Operation.REMOVE_UNINFORMATIVE_FEATURES,
					np.arange(len(scope))[uninformative_features_idx].tolist(),
				)

		if minimalInstances or (no_clusters and no_independencies):
			if multivariate_leaf:
				return Operation.CREATE_LEAF, None
			else:
				return Operation.NAIVE_FACTORIZATION, None

		if no_independencies:
			if param is None:
				return Operation.SPLIT_ROWS, None
			else:
				return Operation.SPLIT_ROWS, param

		if no_clusters:
			if param is None:
				return Operation.SPLIT_COLUMNS, None
			else:
				return Operation.SPLIT_COLUMNS, param

		if is_first:
			if cluster_first:
				if param is None:
					return Operation.SPLIT_ROWS, None
				else:
					return Operation.SPLIT_ROWS, param
			else:
				if param is None:
					return Operation.SPLIT_COLUMNS, None
				else:
					return Operation.SPLIT_COLUMNS, param

		if param is None:
			return Operation.SPLIT_COLUMNS, None
		else:
			return Operation.SPLIT_COLUMNS, param

	return next_operation


def default_slicer(data, cols, num_cond_cols=None):
	if num_cond_cols is None:
		if len(cols) == 1:
			return data[:, cols[0]].reshape((-1, 1))

		return data[:, cols]
	else:
		return np.concatenate((data[:, cols], data[:, -num_cond_cols:]), axis=1)





class AnytimeSPN:


	def __init__(self,
		split_rows,
		split_cols,
		create_leaf,
		next_operation=get_next_operation(),		
		data_slicer=default_slicer,
	):
		

		self.split_rows = split_rows
		self.split_cols = split_cols
		self.create_leaf = create_leaf
		self.next_operation = next_operation
		self.data_slicer = data_slicer

		self.root = Product()
		self.root.children.append(None)
		self.id = 0
		self.llikelihood = list()
		self.spns = list()


	def learn_structure(self, 
		dataset, 
		ds_context,
		initial_scope=None,
	):

		assert dataset is not None
		assert ds_context is not None
		assert self.split_rows is not None
		assert self.split_cols is not None
		assert self.create_leaf is not None
		assert self.next_operation is not None

		if initial_scope is None:
			initial_scope = list(range(dataset.shape[1]))
			num_conditional_cols = None
		elif len(initial_scope) < dataset.shape[1]:
			num_conditional_cols = dataset.shape[1] - len(initial_scope)
		else:
			num_conditional_cols = None
			assert len(initial_scope) >= dataset.shape[1], "check initial scope: %s" % initial_scope


		tasks = deque()
		tasks.append((dataset, self.root, 0, initial_scope, False, False, None, False))
		naiveFactor = 0

		while True:

			if len(tasks)==0:
				break
		
			#print("\n\n\n",tasks)
			#input()
			#Normal executions
			if naiveFactor == 0:
				local_data, parent, children_pos, scope, no_clusters, no_independencies, param, leaf = tasks.popleft()
				operation, op_params = self.next_operation(
					local_data,
					scope,
					self.create_leaf,
					no_clusters=no_clusters,
					no_independencies=no_independencies,
					is_first=(parent is self.root),
					param = param
				)
				
			#Naive Factorize subtrees
			else:
				local_data, parent, children_pos, scope, no_clusters, no_independencies, param, leaf = tasks.pop()
				if leaf:
					operation = Operation.CREATE_LEAF
				else:
					operation = Operation.NAIVE_FACTORIZATION
			print(operation)
			#print("Factor:",naiveFactor)
			logging.debug("OP: {} on slice {} (remaining tasks {})".format(operation, local_data.shape, len(tasks)))

			if operation == Operation.REMOVE_UNINFORMATIVE_FEATURES:
				node = Product()
				node.scope.extend(scope)
				parent.children[children_pos] = node

				rest_scope = set(range(len(scope)))
				for col in op_params:
					rest_scope.remove(col)
					node.children.append(None)
					tasks.append(
						(
							self.data_slicer(local_data, [col], num_conditional_cols),
							node,
							len(node.children) - 1,
							[scope[col]],
							True,
							True,
							None,
							False
						)
					)

				next_final = False

				if len(rest_scope) == 0:
					continue
				elif len(rest_scope) == 1:
					next_final = True

				node.children.append(None)
				c_pos = len(node.children) - 1

				rest_cols = list(rest_scope)
				rest_scope = [scope[col] for col in rest_scope]

				tasks.append(
					(
						self.data_slicer(local_data, rest_cols, num_conditional_cols),
						node,
						c_pos,
						rest_scope,
						next_final,
						next_final,
						None,
						False
					)
				)

				continue

			elif operation == Operation.SPLIT_ROWS:
				
				#Default k
				k = 2
				#Get the k value for next round of XMeans
				if op_params is not None:
					k = op_params
					
				split_start_t = perf_counter()
				
				#Get the new K value and dataslices
				newk, data_slices = self.split_rows(local_data, ds_context, scope, k)
				print("K",k)
				print("NEw K",newk)
				print("Slices",len(data_slices))
				split_end_t = perf_counter()
				logging.debug(
					"\t\tfound {} row clusters (in {:.5f} secs)".format(len(data_slices), split_end_t - split_start_t)
				)

				if len(data_slices) == 1:
					tasks.append((local_data, parent, children_pos, scope, True, False, None, False))
					continue
				
				# If K can be increased, find the clusters again in next iteration
				if k < newk:
					tasks.appendleft((local_data, parent, children_pos, scope, False, True, newk, False))

				#Create sum node
				node = Sum()
				node.scope.extend(scope)
				parent.children[children_pos] = node
				# assert parent.scope == node.scope

				#Add branches to the sum node
				for data_slice, scope_slice, proportion in data_slices:
					assert isinstance(scope_slice, list), "slice must be a list"

					node.children.append(None)
					node.weights.append(proportion)
					tasks.append((data_slice, node, len(node.children) - 1, scope, False, False, None, False))
					

				if k == newk:
					i = 0
					for data_slice, scope_slice, proportion in data_slices:
						tasks.append((data_slice, node, i, scope, False, False, None, False))
						i+=1
				# If newk > k, naiveFactorize subtrees
				if newk > k:
					naiveFactor = len(node.children)
				continue

			elif operation == Operation.SPLIT_COLUMNS:

				#Default k
				n = int(local_data.shape[1]**0.5)
				#Get the k value for next round of variable splitting
				if op_params is not None:
					n = op_params
				split_start_t = perf_counter()
				data_slices = self.split_cols(local_data, ds_context, scope, n)
				split_end_t = perf_counter()
				logging.debug(
					"\t\tfound {} col clusters (in {:.5f} secs)".format(len(data_slices), split_end_t - split_start_t)
				)
				print(len(data_slices))
				if len(data_slices) == 1:
					tasks.append((local_data, parent, children_pos, scope, False, True, None, False))
					assert np.shape(data_slices[0][0]) == np.shape(local_data)
					assert data_slices[0][1] == scope
					continue

				if n < local_data.shape[1]:
					tasks.appendleft((local_data, parent, children_pos, scope, True, False, n+1, False))

				node = Product()
				node.scope.extend(scope)
				parent.children[children_pos] = node

				for data_slice, scope_slice, _ in data_slices:
					assert isinstance(scope_slice, list), "slice must be a list"
					print(scope_slice)
					node.children.append(None)
					if len(scope_slice) > 1:
						tasks.append((data_slice, node, len(node.children) - 1, scope_slice, False, False, None, False))
					else:
						tasks.append((data_slice, node, len(node.children) - 1, scope_slice, False, False, None, True))
				print(node.children)

				if n == local_data.shape[1]:
					i=0
					for data_slice, scope_slice, _ in data_slices:
						if len(scope_slice) > 1:
							tasks.append((data_slice, node, i, scope_slice, False, False, None, False))
						else:
							tasks.append((data_slice, node, i, scope_slice, False, False, None, True))
						i+=1
				if n < local_data.shape[1]:
					naiveFactor = len(node.children)
				continue

			elif operation == Operation.NAIVE_FACTORIZATION:
				node = Product()
				node.scope.extend(scope)
				parent.children[children_pos] = node

				local_tasks = []
				local_children_params = []
				split_start_t = perf_counter()
				for col in range(len(scope)):
					node.children.append(None)
					# tasks.append((data_slicer(local_data, [col], num_conditional_cols), node, len(node.children) - 1, [scope[col]], True, True))
					local_tasks.append(len(node.children) - 1)
					child_data_slice = self.data_slicer(local_data, [col], num_conditional_cols)
					local_children_params.append((child_data_slice, ds_context, [scope[col]]))

				result_nodes = pool.starmap(self.create_leaf, local_children_params)
				# result_nodes = []
				# for l in tqdm(local_children_params):
				#    result_nodes.append(create_leaf(*l))
				# result_nodes = [create_leaf(*l) for l in local_children_params]
				for child_pos, child in zip(local_tasks, result_nodes):
					node.children[child_pos] = child

				split_end_t = perf_counter()

				logging.debug(
					"\t\tnaive factorization {} columns (in {:.5f} secs)".format(len(scope), split_end_t - split_start_t)
				)
				if naiveFactor == 1:
					spn = Copy(self.return_spn())
					self.id += 1
					print(f"\n\n\n\nSPN {self.id} created\n\n\n\n")
					print(get_structure_stats(spn))
					self.spns.append(spn)
					
					
				naiveFactor = max(0, naiveFactor-1)

				continue

			elif operation == Operation.CREATE_LEAF:
				leaf_start_t = perf_counter()
				node = self.create_leaf(local_data, ds_context, scope)
				parent.children[children_pos] = node
				leaf_end_t = perf_counter()

				logging.debug(
					"\t\t created leaf {} for scope={} (in {:.5f} secs)".format(
						node.__class__.__name__, scope, leaf_end_t - leaf_start_t
					)
				)

				if naiveFactor == 1:
					spn = Copy(self.return_spn())
					self.id += 1
					print(f"\n\n\n\nSPN {self.id} created\n\n\n\n")
					print(get_structure_stats(spn))
					self.spns.append(spn)
					
					
				naiveFactor = max(0, naiveFactor-1)

			else:
				raise Exception("Invalid operation: " + operation)

		spn = Copy(self.return_spn())
		self.id += 1
		print(f"\n\n\n\nSPN {self.id} created\n\n\n\n")
		print(get_structure_stats(spn))
		self.spns.append(spn)
		return self.spns
		
	def return_spn(self):
		node = Copy(self.root.children[0])
		assign_ids(node)
		valid, err = is_valid(node)
		assert valid, "invalid spn: " + err
		node = Prune(node)
		valid, err = is_valid(node)
		assert valid, "invalid spn: " + err

		return node
