
from spn.structure.Base import Sum, Product, Max
from spn.structure.Base import assign_ids, rebuild_scopes_bottom_up
from spn.algorithms.LearningWrappers import learn_parametric_aspmn, learn_mspn_for_aspmn
from spn.algorithms.splitting.RDC import get_split_cols_distributed_RDC_py1, get_split_cols_RDC_py, get_split_cols_single_RDC_py
from spn.algorithms.SPMNHelper import *
from spn.algorithms.MEU import meu
from spn.algorithms.Inference import log_likelihood
from spn.algorithms.Statistics import get_structure_stats_dict
from spn.io.Graphics import plot_spn
from spn.io.ProgressBar import printProgressBar
from spn.data.simulator import get_env
from spn.algorithms.MEU import best_next_decision
import multiprocessing
import logging
import numpy as np
import matplotlib.pyplot as plt
from os import path as pth
import sys, os
import math
import pickle

from spn.algorithms.TransformStructure import Prune


class Anytime_SPMN:

	def __init__(self, dataset, output_path, partial_order, decision_nodes, utility_node, feature_names, feature_labels,
			meta_types, cluster_by_curr_information_set=False, util_to_bin=False):

		self.dataset = dataset
		self.params = SPMNParams(
				partial_order,
				decision_nodes,
				utility_node,
				feature_names,
				feature_labels,
				meta_types,
				util_to_bin
			)
		self.op = 'Any'
		self.cluster_by_curr_information_set = cluster_by_curr_information_set
		self.spmn = None

		self.vars = len(feature_labels)

		self.plot_path = f"{output_path}/{dataset}"


		if not pth.exists(self.plot_path):
			try:
				os.makedirs(self.plot_path)
			except OSError:
				print ("Creation of the directory %s failed" % self.plot_path)
				sys.exit()


	def set_next_operation(self, next_op):
		self.op = next_op

	def get_curr_operation(self):
		return self.op

	def __learn_spmn_structure(self, remaining_vars_data, remaining_vars_scope,
							   curr_information_set_scope, index):

		logging.info(f'start of new recursion in __learn_spmn_structure method of SPMN')
		logging.debug(f'remaining_vars_scope: {remaining_vars_scope}')
		logging.debug(f'curr_information_set_scope: {curr_information_set_scope}')

		# rest set is remaining variables excluding the variables in current information set
		rest_set_scope = [var_scope for var_scope in remaining_vars_scope if
						  var_scope not in curr_information_set_scope]
		logging.debug(f'rest_set_scope: {rest_set_scope}')

		scope_index = sum([len(x) for x in self.params.partial_order[:index]])
		next_scope_index = sum([len(x) for x in self.params.partial_order[:index + 1]])

		if remaining_vars_scope == curr_information_set_scope:
			# this is last information set in partial order. Base case of recursion

			# test if current information set is a decision node
			if self.params.partial_order[index][0] in self.params.decision_nodes:
				raise Exception(f'last information set of partial order either contains random '
								f'and utility variables or just a utility variable. '
								f'This contains decision variable: {self.params.partial_order[index][0]}')

			else:
				# contains just the random and utility variables

				logging.info(f'at last information set of this recursive call: {curr_information_set_scope}')
				ds_context_last_information_set = get_ds_context(remaining_vars_data,
																 remaining_vars_scope, self.params)

				if self.params.util_to_bin:

					last_information_set_spn = learn_parametric_aspmn(remaining_vars_data,
																ds_context_last_information_set,
																n=self.n,
																k_limit=self.limit,
																min_instances_slice=20,
																initial_scope=remaining_vars_scope)

				else:

					last_information_set_spn = learn_mspn_for_aspmn(remaining_vars_data,
																   ds_context_last_information_set,
																   n=self.n,
																   k_limit=self.limit,
																   min_instances_slice=20,
																   initial_scope=remaining_vars_scope)

			logging.info(f'created spn at last information set')
			return last_information_set_spn

		# test for decision node. test if current information set is a decision node
		elif self.params.partial_order[index][0] in self.params.decision_nodes:

			decision_node = self.params.partial_order[index][0]

			logging.info(f'Encountered Decision Node: {decision_node}')

			# cluster the data from remaining variables w.r.t values of decision node
			clusters_on_next_remaining_vars, dec_vals = anytime_split_on_decision_node(remaining_vars_data, int(self.d))
			#clusters_on_next_remaining_vars, dec_vals = split_on_decision_node(remaining_vars_data)

			decision_node_children_spns = []
			index += 1

			next_information_set_scope = np.array(range(next_scope_index, next_scope_index +
														len(self.params.partial_order[index]))).tolist()

			next_remaining_vars_scope = rest_set_scope
			self.set_next_operation('Any')

			logging.info(f'split clusters based on decision node values')
			for cluster_on_next_remaining_vars in clusters_on_next_remaining_vars:

				decision_node_children_spns.append(self.__learn_spmn_structure(cluster_on_next_remaining_vars,
																			   next_remaining_vars_scope,
																			   next_information_set_scope, index
																			   ))

			decision_node_spn_branch = Max(dec_idx=scope_index, dec_values=dec_vals,
										   children=decision_node_children_spns, feature_name=decision_node)

			assign_ids(decision_node_spn_branch)
			rebuild_scopes_bottom_up(decision_node_spn_branch)
			logging.info(f'created decision node')
			return decision_node_spn_branch

		# testing for independence
		else:

			curr_op = self.get_curr_operation()
			logging.debug(f'curr_op at prod node (independence test): {curr_op}')

			if curr_op != 'Sum':    # fails if correlated variable set found in previous recursive call.
									# Without this condition code keeps looping at this stage

				ds_context = get_ds_context(remaining_vars_data, remaining_vars_scope, self.params)

				#split_cols = get_split_cols_single_RDC_py(rand_gen=None, ohe=False, n_jobs=-1, n=round(self.n))
				split_cols = get_split_cols_distributed_RDC_py1(rand_gen=None, ohe=False, n_jobs=-1, n=round(self.n))
				data_slices_prod = split_cols(remaining_vars_data, ds_context, remaining_vars_scope, rest_set_scope)
				#split_cols = get_split_cols_RDC_py()
				#data_slices_prod = split_cols(remaining_vars_data, ds_context, remaining_vars_scope)

				logging.debug(f'{len(data_slices_prod)} slices found at data_slices_prod: ')

				prod_children = []
				next_remaining_vars_scope = []
				independent_vars_scope = []

				'''
				print('\n\nProduct:')
				for cluster, scope, weight in data_slices_prod:
					print(scope)
				'''

				for correlated_var_set_cluster, correlated_var_set_scope, weight in data_slices_prod:

					if any(var_scope in correlated_var_set_scope for var_scope in rest_set_scope):

						next_remaining_vars_scope.extend(correlated_var_set_scope)

					else:
						# this variable set of current information set is
						# not correlated to any variable in the rest set

						logging.info(f'independent variable set found: {correlated_var_set_scope}')

						ds_context_prod = get_ds_context(correlated_var_set_cluster,
														 correlated_var_set_scope, self.params)

						if self.params.util_to_bin:

							independent_var_set_prod_child = learn_parametric_aspmn(correlated_var_set_cluster,
																			  ds_context_prod,
																			  n=self.n,
																			  k_limit=self.limit,
																			  min_instances_slice=20,
																			  initial_scope=correlated_var_set_scope)

						else:

							independent_var_set_prod_child = learn_mspn_for_aspmn(correlated_var_set_cluster,
																				 ds_context_prod,
																				 n=self.n,
																				 k_limit=self.limit,
																				 min_instances_slice=20,
																				 initial_scope=correlated_var_set_scope)
						independent_vars_scope.extend(correlated_var_set_scope)
						prod_children.append(independent_var_set_prod_child)

				logging.info(f'correlated variables over entire remaining variables '
							 f'at prod, passed for next recursion: '
							 f'{next_remaining_vars_scope}')
				# check if all variables in current information set are consumed
				if all(var_scope in independent_vars_scope for var_scope in curr_information_set_scope):

					index += 1
					next_information_set_scope = np.array(range(next_scope_index, next_scope_index +
																len(self.params.partial_order[index]))).tolist()

					# since current information set is totally consumed
					next_remaining_vars_scope = rest_set_scope

				else:
					# some variables in current information set still remain
					index = index

					next_information_set_scope = set(curr_information_set_scope) - set(independent_vars_scope)
					next_remaining_vars_scope = next_information_set_scope | set(rest_set_scope)

					# convert unordered sets of scope to sorted lists to keep in sync with partial order
					next_information_set_scope = sorted(list(next_information_set_scope))
					next_remaining_vars_scope = sorted(list(next_remaining_vars_scope))
				self.set_next_operation('Sum')

				next_remaining_vars_data = column_slice_data_by_scope(remaining_vars_data,
																	  remaining_vars_scope,
																	  next_remaining_vars_scope)

				logging.info(
					f'independence test completed for current information set {curr_information_set_scope} '
					f'and rest set {rest_set_scope} ')

				remaining_vars_prod_child = self.__learn_spmn_structure(next_remaining_vars_data,
																		next_remaining_vars_scope,
																		next_information_set_scope,
																		index)

				prod_children.append(remaining_vars_prod_child)

				product_node = Product(children=prod_children)
				assign_ids(product_node)
				rebuild_scopes_bottom_up(product_node)

				logging.info(f'created product node')
				return product_node

			# Cluster the data
			else:

				curr_op = self.get_curr_operation()
				logging.debug(f'curr_op at sum node (cluster test): {curr_op}')

				split_rows = get_split_rows_XMeans(limit=self.limit)    # from SPMNHelper.py
				#split_rows = get_split_rows_KMeans()

				if self.cluster_by_curr_information_set:

					curr_information_set_data = column_slice_data_by_scope(remaining_vars_data,
																		   remaining_vars_scope,
																		   curr_information_set_scope)

					ds_context_sum = get_ds_context(curr_information_set_data, curr_information_set_scope, self.params)
					data_slices_sum, km_model = split_rows(curr_information_set_data, ds_context_sum,
														   curr_information_set_scope)

					logging.info(f'split clusters based on current information set {curr_information_set_scope}')

				else:
					# cluster on whole remaining variables
					ds_context_sum = get_ds_context(remaining_vars_data, remaining_vars_scope, self.params)
					data_slices_sum, km_model = split_rows(remaining_vars_data, ds_context_sum, remaining_vars_scope)

					logging.info(f'split clusters based on whole remaining variables {remaining_vars_scope}')

				sum_node_children = []
				weights = []
				index = index
				logging.debug(f'{len(data_slices_sum)} clusters found at data_slices_sum')



				cluster_num = 0
				labels_array = km_model.labels_
				logging.debug(f'cluster labels of rows: {labels_array} used to cluster data on '
							  f'total remaining variables {remaining_vars_scope}')

				for cluster, scope, weight in data_slices_sum:

					self.set_next_operation("Prod")

					# cluster whole remaining variables based on clusters formed.
					# below methods are useful if clusters were formed on just the current information set

					cluster_indices = get_row_indices_of_cluster(labels_array, cluster_num)
					cluster_on_remaining_vars = row_slice_data_by_indices(remaining_vars_data, cluster_indices)

					# logging.debug(np.array_equal(cluster_on_remaining_vars, cluster ))

					sum_node_children.append(
						self.__learn_spmn_structure(cluster_on_remaining_vars, remaining_vars_scope,
													curr_information_set_scope, index))

					weights.append(weight)

					cluster_num += 1

				sum_node = Sum(weights=weights, children=sum_node_children)

				assign_ids(sum_node)
				rebuild_scopes_bottom_up(sum_node)
				logging.info(f'created sum node')
				return sum_node


	def get_loglikelihood(self, instance):
		test_data = np.array(instance).reshape(-1, len(self.params.feature_names))
		return log_likelihood(self.spmn, test_data)[0][0]


	def get_reward(self, ids):

		
		state = self.env.reset()
		while(True):
			output = best_next_decision(self.spmn, state)
			action = output[0][0]
			state, reward, done = self.env.step(action)
			if done:
				
				return reward

	def get_reward1(self, ids):

		policy = ""
		state = self.env.reset()
		while(True):
			output = best_next_decision(self.spmn, state)
			action = output[0][0]
			policy += f"{action}  "
			state, reward, done = self.env.step(action)
			'''
			if action==1:
				print(state)
				#
			'''
			if done:
				#return reward
				return policy

	def learn_aspmn(self, train, test, k=None):
		"""
		:param
		:return: learned spmn
		"""
		
		original_stats = {
			'Export_Textiles': {"ll" : -1.0890750655173789, "meu" : 1722313.8158882717, 'nodes' : 38, 'reward':1721301.8260000004, 'dev':3861.061525772288},
			'Test_Strep': {"ll" : -0.9130071749277912, "meu" : 54.9416526618876, 'nodes' : 130, 'reward':54.91352060400901, 'dev':0.013189836549851251},
			'LungCancer_Staging': {"ll" : -1.1489156814245234, "meu" : 3.138664586296027, 'nodes' : 312, 'reward':3.108005299999918, 'dev':0.011869627022775012},
			'HIV_Screening': {"ll" : -0.6276399171508842, "meu" : 42.582734183407034, 'nodes' : 112, 'reward':42.559788119992646, 'dev':0.06067708771159484},
			'Computer_Diagnostician': {"ll" : -0.9011245432112749, "meu" : -208.351, 'nodes' : 56, 'reward':-210.15520000000004, 'dev':0.3810022440878799},
			'Powerplant_Airpollution': {"ll" : -1.0796885930912947, "meu" : -2756263.244346315, 'nodes' : 38, 'reward':-2759870.4, 'dev':6825.630813338794}
		}

		optimal_meu = {
			'Export_Textiles' : 1721300,
			'Computer_Diagnostician': -210.13,
			'Powerplant_Airpollution': -2760000,
			'HIV_Screening': 42.5597,
			'Test_Strep': 54.9245,
			'LungCancer_Staging': 3.12453
		}

		random_reward = {
			'Export_Textiles' : {'reward': 1300734.02, 'dev':7087.350616838437},
			'Computer_Diagnostician': {'reward': -226.666, 'dev':0.37205611135956335},
			'Powerplant_Airpollution': {'reward': -3032439.0, 'dev':7870.276615214995},
			'HIV_Screening': {'reward': 42.3740002199867, 'dev':0.07524234474837802},
			'Test_Strep': {'reward': 54.89614493400057, 'dev':0.012847272731391593},
			'LungCancer_Staging': {'reward': 2.672070640000026, 'dev':0.007416967451081523},
		}

		
		
		trials = 500000
		interval = 10000
		batches = 25
		interval_count = int(trials/interval)

		avg_rewards = [list() for i in range(int(trials/interval))]
		reward_dev = [list() for i in range(int(trials/interval))]

	
		
		avg_ll = list()
		ll_dev = list()
		meus = list()
		nodes = list()
		#avg_rewards = list()
		#reward_dev = list()
		past3 = list()

		self.env = get_env(self.dataset)
		
		limit = 2 
		n = int(self.vars**0.5)
		#n= self.vars
		step = 0 
		step = (self.vars - (self.vars**0.5) + 1)/10
		d = 2
		d_max = 4
		d_step = (d_max - d + 1)/10

		if not pth.exists(f"{self.plot_path}/models"):
			try:
				os.makedirs(f"{self.plot_path}/models")
			except OSError:
				print ("Creation of the directory failed")
				sys.exit()

		i = 0
		while(True):

			index = 0
			print(f"\nIteration: {i}\n")
			
			curr_information_set_scope = np.array(range(len(self.params.partial_order[0]))).tolist()
			remaining_vars_scope = np.array(range(len(self.params.feature_names))).tolist()
			self.set_next_operation('Any')
			self.limit = limit 
			self.n = n  
			self.d = d

			print("\nStart Learning...")
			spmn = self.__learn_spmn_structure(train, remaining_vars_scope, curr_information_set_scope, index)
			print("SPMN Learned")
			#spmn = Prune(spmn)
			self.spmn = spmn

			file = open(f"{self.plot_path}/models/spn_{i}.pkle",'wb')
			pickle.dump(self.spmn, file)
			file.close()

			nodes.append(get_structure_stats_dict(spmn)["nodes"])

			
			#plot_spn(spmn, f'{self.plot_path}/spmn{i}.pdf', feature_labels=self.params.feature_labels)
			
			
			#try:
			
			total_ll = 0
			trials1 = test.shape[0]
			batch_size = int(trials1 / 10)
			batch = list()
			pool = multiprocessing.Pool()

			
			
			for b in range(10):
				test_slice = test[b*batch_size:(b+1)*batch_size]
				lls = pool.map(self.get_loglikelihood, test_slice)
				total_ll = sum(lls)
				batch.append(total_ll/batch_size)
				printProgressBar(b+1, 10, prefix = f'Log Likelihood Evaluation :', suffix = 'Complete', length = 50)
			
			'''
			for j, instance in enumerate(test):
				test_data = np.array(instance).reshape(-1, len(self.params.feature_names))
				total_ll += log_likelihood(spmn, test_data)[0][0]
				if (j+1) % batch_size == 0:
					batch.append(total_ll/batch_size)
					total_ll = 0
				printProgressBar(j+1, len(test), prefix = f'Log Likelihood Evaluation :', suffix = 'Complete', length = 50)
			'''

			avg_ll.append(np.mean(batch))
			ll_dev.append(np.std(batch))
			


			test_data = [[np.nan]*len(self.params.feature_names)]
			m = meu(spmn, test_data)
			meus.append(m[0])

			plt.close()
			
			plt.plot(meus, marker="o", label="Anytime")
			#plt.plot([optimal_meu[self.dataset]]*len(meus), linewidth=3, color ="lime", label="Optimal MEU")
			#plt.plot([original_stats[self.dataset]["meu"]]*len(meus), linestyle="dashed", color ="red", label="LearnSPMN")
			plt.title(f"{self.dataset} MEU")
			plt.legend()
			if k is None:
				plt.savefig(f"{self.plot_path}/meu.png", dpi=100)
			else:
				plt.savefig(f"{self.plot_path}/{k}/meu.png", dpi=100)
			plt.close()
			
			
			
			total_reward = 0
			rewards = list()

			
			from collections import Counter
			pool = multiprocessing.Pool()
			for inter in range(interval_count):
				
				for y in range(batches):
					ids = [None for x in range(int(interval/batches))]
					cur = pool.map(self.get_reward, ids)
					rewards += cur
					z = (inter*batches) + y + 1
					printProgressBar(z, interval_count*batches, prefix = f'Average Reward Evaluation :', suffix = 'Complete', length = 50)

				
				batch = list()
				batch_size = int(len(rewards) / batches)
				for l in range(batches):
					m = l*batch_size
					batch.append(sum(rewards[m:m+batch_size]) / batch_size)
				

				avg_rewards[inter].append(np.mean(batch))
				reward_dev[inter].append(np.std(batch))

				plt.close()
				'''
				rand_reward = np.array([random_reward[self.dataset]["reward"]]*len(avg_rewards[inter]))
				dev = np.array([random_reward[self.dataset]["dev"]]*len(avg_rewards[inter]))
				plt.fill_between(np.arange(len(avg_rewards[inter])), rand_reward-dev, rand_reward+dev, alpha=0.1, color="lightgrey")
				plt.plot(rand_reward, linestyle="dashed", color ="grey", label="Random Policy")

				original_reward = np.array([original_stats[self.dataset]["reward"]]*len(avg_rewards[inter]))
				dev = np.array([original_stats[self.dataset]["dev"]]*len(avg_rewards[inter]))
				plt.fill_between(np.arange(len(avg_rewards[inter])), original_reward-dev, original_reward+dev, alpha=0.3, color="red")
				plt.plot([optimal_meu[self.dataset]]*len(avg_rewards[inter]), linewidth=3, color ="lime", label="Optimal MEU")
				plt.plot(original_reward, linestyle="dashed", color ="red", label="LearnSPMN")
				'''
				plt.errorbar(np.arange(len(avg_rewards[inter])), avg_rewards[inter], yerr=reward_dev[inter], marker="o", label="Anytime")
				plt.title(f"{self.dataset} Average Rewards")
				plt.legend()
				plt.savefig(f"{self.plot_path}/rewards_trend_{(inter+1)*interval}.png", dpi=100)
				plt.close()

				f = open(f"{self.plot_path}/stats_trends.txt", "w")

				f.write(f"\n{self.dataset}")

				for x in range(interval_count):

					f.write(f"\n\n\tAverage Rewards {(x+1)*interval}: {avg_rewards[x]}")
					f.write(f"\n\tDeviation {(x+1)*interval}: {reward_dev[x]}")

				f.close()

				
			
			
			print("\n\n\n\n\n")
			print(f"X-Means Limit: {limit}, \tVariables for splitting: {round(n)}")
			print("#Nodes: ",nodes[-1])
			print("Log Likelihood: ",avg_ll[-1])
			print("Log Likelihood Deviation: ",ll_dev[-1])
			print("MEU: ",meus[-1])
			print("Average rewards: ",avg_rewards[-1][-1])
			print("Deviation: ",reward_dev[-1][-1])
			print(nodes)
			print(meus)
			print("\n\n\n\n\n")
			
			
			plt.close()
			# plot line 
			#plt.plot([original_stats[self.dataset]["ll"]]*len(avg_ll), linestyle="dashed", color ="red", label="LearnSPMN")
			plt.errorbar(np.arange(len(avg_ll)), avg_ll, yerr=ll_dev, marker="o", label="Anytime")
			plt.title(f"{self.dataset} Log Likelihood")
			plt.legend()
			if k is None:
				plt.savefig(f"{self.plot_path}/ll.png", dpi=100)
			else:
				plt.savefig(f"{self.plot_path}/{k}/ll.png", dpi=100)
			'''
			plt.close()
			
			plt.plot(meus, marker="o", label="Anytime")
			#plt.plot([optimal_meu[self.dataset]]*len(meus), linewidth=3, color ="lime", label="Optimal MEU")
			plt.plot([original_stats[self.dataset]["meu"]]*len(meus), linestyle="dashed", color ="red", label="LearnSPMN")
			plt.title(f"{self.dataset} MEU")
			plt.legend()
			if k is None:
				plt.savefig(f"{self.plot_path}/meu1.png", dpi=100)
			else:
				plt.savefig(f"{self.plot_path}/{k}/meu1.png", dpi=100)
			'''
			plt.close()

			plt.plot(nodes, marker="o", label="Anytime")
			#plt.plot([original_stats[self.dataset]["nodes"]]*len(nodes), linestyle="dashed", color ="red", label="LearnSPMN")
			plt.title(f"{self.dataset} Nodes")
			plt.legend()
			if k is None:
				plt.savefig(f"{self.plot_path}/nodes.png", dpi=100)
			else:
				plt.savefig(f"{self.plot_path}/nodes.png", dpi=100)
			plt.close()
			
			'''
			original_reward = np.array([original_stats[self.dataset]["reward"]]*len(avg_rewards))
			dev = np.array([original_stats[self.dataset]["dev"]]*len(avg_rewards))
			plt.fill_between(np.arange(len(avg_rewards)), original_reward-dev, original_reward+dev, alpha=0.3, color="red")
			plt.plot(original_reward, linestyle="dashed", color ="red", label="LearnSPMN")
			plt.errorbar(np.arange(len(avg_rewards)), avg_rewards, yerr=reward_dev, marker="o", label="Anytime")
			plt.title(f"{self.dataset} Average Rewards")
			plt.legend()
			if k is None:
				plt.savefig(f"{self.plot_path}/rewards.png", dpi=100)
			else:
				plt.savefig(f"{self.plot_path}/rewards.png", dpi=100)
			plt.close()


			
			'''
			

			
			f = open(f"{self.plot_path}/stats.txt", "w") if k is None else open(f"{self.plot_path}/{k}/stats.txt", "w")

			f.write(f"\n{self.dataset}")
			f.write(f"\n\tLog Likelihood : {avg_ll}")
			f.write(f"\n\tLog Likelihood Deviation: {ll_dev}")
			f.write(f"\n\tMEU : {meus}")
			f.write(f"\n\tNodes : {nodes}")
			f.write(f"\n\tAverage Rewards : {avg_rewards[-1]}")
			f.write(f"\n\tRewards Deviation : {reward_dev[-1]}")
			f.close()
			
			#except:
				#pass
			
				
			if n>=self.vars and round(np.std(past3), 3) <= 0.001:
				break


			i += 1
			limit += 1
			d += d_step
			n = n+step
			#if step == 0:
			#	step = 1

		'''
		stats = {"ll" : avg_ll,
				"ll_dev": ll_dev,
				"meu" : meus,
				"nodes" : nodes,
				"reward" : avg_rewards,
				"deviation" : reward_dev
				}
		'''
		# Prune(self.spmn)
		return self.spmn #, stats







class SPMNParams:

	def __init__(self, partial_order, decision_nodes, utility_nodes, feature_names, feature_labels, meta_types, util_to_bin):

		self.partial_order = partial_order
		self.decision_nodes = decision_nodes
		self.utility_nodes = utility_nodes
		self.feature_names = feature_names
		self.feature_labels = feature_labels
		self.meta_types = meta_types
		self.util_to_bin = util_to_bin
