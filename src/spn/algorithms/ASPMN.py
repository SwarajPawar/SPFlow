
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
import logging
import numpy as np
import matplotlib.pyplot as plt
from os import path as pth
import sys, os
import math

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
			#clusters_on_next_remaining_vars, dec_vals = anytime_split_on_decision_node(remaining_vars_data, self.d)
			clusters_on_next_remaining_vars, dec_vals = split_on_decision_node(remaining_vars_data)

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

	def learn_aspmn(self, train, test, k=None):
		"""
		:param
		:return: learned spmn
		"""
		
		'''
		original_stats_old = {
			'Export_Textiles': {"ll" : -1.0903135560503194, "meu" : 1922639.5, 'nodes' : 22},
			'Test_Strep': {"ll" : -1.1461735112245122, "meu" : 54.92189449375, 'nodes' : 51},
			'LungCancer_Staging': {"ll" : -1.3292497032277288, "meu" : 3.11376125, 'nodes' : 49},
			'HIV_Screening': {"ll" : -0.5943350928785097, "meu" : 42.60624317138454, 'nodes' : 125},
			'Computer_Diagnostician': {"ll" : -0.8912294493362266, "meu" : 242.863042737567, 'nodes' : 50},
			'Powerplant_Airpollution': {"ll" : -1.8151637099020188, "meu" : -2803562.5, 'nodes' : 45}
		}
		'''
		'''
		original_stats_new = {
			'Export_Textiles': {"ll" : -1.0892559429908522, "meu" : 1922275.95, 'nodes' : 22, 'reward':1721469.45},
			'Test_Strep': {"ll" : -0.9112557170813002, "meu" : 54.93760881256758, 'nodes' : 100, 'reward':54.97011839999944},
			'LungCancer_Staging': {"ll" : -1.1515872880624247, "meu" : 3.1526200852839716, 'nodes' : 260, 'reward':3.1738849999999976},
			'HIV_Screening': {"ll" : -0.6189833438168413, "meu" : 42.63750815337698, 'nodes' : 112, 'reward':42.4838739999994},
			'Computer_Diagnostician': {"ll" : -0.892138328151404, "meu" : 244.94, 'nodes' : 47, 'reward':244.955},
			'Powerplant_Airpollution': {"ll" : -1.081424145432235, "meu" : -2726821.30929344245, 'nodes' : 46, 'reward':-2770200.0}
		}
		'''
		
		original_stats = {
			'Export_Textiles': {"ll" : -1.0890750655173789, "meu" : 1722313.8158882717, 'nodes' : 38, 'reward':1716130.8399999999, 'dev':8877.944736840887},
			'Test_Strep': {"ll" : -0.9130071749277912, "meu" : 54.9416526618876, 'nodes' : 130, 'reward':54.93578280000071, 'dev':0.018246756840598732},
			'LungCancer_Staging': {"ll" : -1.1489156814245234, "meu" : 3.138664586296027, 'nodes' : 312, 'reward':3.1265179999999946, 'dev':0.024158974233189766},
			'HIV_Screening': {"ll" : -0.6276399171508842, "meu" : 42.582734183407034, 'nodes' : 112, 'reward':42.64759879999822, 'dev':0.13053757307440556},
			'Computer_Diagnostician': {"ll" : -0.8920749045689644, "meu" : 244.85700000000003, 'nodes' : 47, 'reward':245.04599999999996, 'dev':0.40763218714915067},
			'Powerplant_Airpollution': {"ll" : -1.0796486063753, "meu" : -2756263.244346315, 'nodes' : 46, 'reward':-2750100.0, 'dev':25448.182646310914}
		}

		'''
		max_stats = {
			'Export_Textiles': {"ll" : -1.085894618117626, "meu" : 1722313.8158882714, 'nodes' : 38, 'reward':1734820.15},
			'Test_Strep': {"ll" : -1.326788892481826, "meu" : 54.89817723280146, 'nodes' : 143, 'reward':54.8730194999996},
			'LungCancer_Staging': {"ll" : -2.2087761740149308, "meu" : 2.672605436120087, 'nodes' : 492, 'reward':3.1100099999999955},
			'HIV_Screening': {"ll" : -1.22956668145111, "meu" : 42.413024507034876, 'nodes' : 122, 'reward':42.28838000000008},
			'Computer_Diagnostician': {"ll" : -1.399600088876896, "meu" : 226.26550000000006, 'nodes' : 52, 'reward':209.2775},
			'Powerplant_Airpollution': {"ll" : -1.1907544739362805, "meu" : -3000000.0, 'nodes' : 49, 'reward':-3000000.0}
		}
		'''

		max_stats = {
			'Export_Textiles': {"ll" : -1.0890750655173789, "meu" : 1722313.8158882717, 'nodes' : 38, 'reward':1729827.6},
			'Test_Strep': {"ll" : -0.9130071749277912, "meu" : 54.9416526618876, 'nodes' : 130, 'reward':54.939131999999425},
			'LungCancer_Staging': {"ll" : -1.1489156814245234, "meu" : 3.138664586296027, 'nodes' : 312, 'reward':3.153284999999997},
			'HIV_Screening': {"ll" : -0.6276399171508842, "meu" : 42.582734183407034, 'nodes' : 112, 'reward':42.5504879999994},
			'Computer_Diagnostician': {"ll" : -0.8920749045689644, "meu" : 244.85700000000003, 'nodes' : 47, 'reward':245.27000000000004},
			'Powerplant_Airpollution': {"ll" : -1.0796486063753, "meu" : -2756263.244346315, 'nodes' : 46, 'reward':-2755600.0}
		}
		
		trials = 150000
		interval = 10000
		batches = 10


		avg_rewards = [list() for i in range(int(trials/interval))]
		reward_dev = [list() for i in range(int(trials/interval))]

	
		'''
		avg_ll = list()
		ll_dev = list()
		meus = list()
		nodes = list()
		avg_rewards = list()
		reward_dev = list()
		past3 = list()
		'''
		
		limit = 2 
		n = int(self.vars**0.5)
		#n= self.vars
		step = (self.vars - (self.vars**0.5) + 1)/10
		d = 2

		if k is not None:
			if not pth.exists(f"{self.plot_path}/{k}"):
				try:
					os.makedirs(f"{self.plot_path}/{k}")
				except OSError:
					print ("Creation of the directory %s failed" % f"{self.plot_path}/{k}")
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
			print("done")
			#spmn = Prune(spmn)
			self.spmn = spmn


			env = get_env(self.dataset)
			total_reward = 0
			rewards = list()
			
			inter = 0
			for z in range(trials):
				
				state = env.reset()
				while(True):
					output = best_next_decision(spmn, state)
					action = output[0][0]
					state, reward, done = env.step(action)
					if done:
						rewards.append(reward)
						break
				if (z+1) % interval == 0:
					batch = list()
					batch_size = int(z / batches)
					for l in range(batches):
						m = l*batch_size
						batch.append(sum(rewards[m:m+batch_size]) / batch_size)
					
					avg_rewards[inter].append(np.mean(batch))
					reward_dev[inter].append(np.std(batch))

					original_reward = np.array([original_stats[self.dataset]["reward"]]*len(avg_rewards[inter]))
					dev = np.array([original_stats[self.dataset]["dev"]]*len(avg_rewards[inter]))
					plt.plot(original_reward, linestyle="dotted", color ="red", label="LearnSPMN")
					plt.fill_between(np.arange(len(avg_rewards[inter])), original_reward-dev, original_reward+dev, alpha=0.3, color="red")
					plt.errorbar(np.arange(len(avg_rewards[inter])), avg_rewards[inter], yerr=reward_dev[inter], marker="o", label="Anytime")
					plt.title(f"{self.dataset} Average Rewards")
					plt.legend()
					plt.savefig(f"{self.plot_path}/rewards_trend_{(inter+1)*interval}.png", dpi=100)
					plt.close()

					f = open(f"{self.plot_path}/stats_trends.txt", "w")

					f.write(f"\n{self.dataset}")

					for x in range(int(trials/interval)):

						f.write(f"\n\n\tAverage Rewards {(x+1)*interval}: {avg_rewards[x]}")
						f.write(f"\n\tDeviation {(x+1)*interval}: {reward_dev[x]}")

					f.close()

					inter += 1

				printProgressBar(z+1, trials, prefix = f'Average Reward Evaluation :', suffix = 'Complete', length = 50)

			'''

			nodes.append(get_structure_stats_dict(spmn)["nodes"])

			
			if k is None:
				plot_spn(spmn, f'{self.plot_path}/spmn{i}.pdf', feature_labels=self.params.feature_labels)
			else:
				plot_spn(spmn, f'{self.plot_path}/{k}/spmn{i}.pdf', feature_labels=self.params.feature_labels)
			
			
			#try:
			total_ll = 0
			trials = test.shape[0]
			batch_size = trials / 10
			batch = list()
			for j, instance in enumerate(test):
				test_data = np.array(instance).reshape(-1, len(self.params.feature_names))
				total_ll += log_likelihood(spmn, test_data)[0][0]
				if (j+1) % batch_size == 0:
					batch.append(total_ll/batch_size)
					total_ll = 0
				printProgressBar(j+1, len(test), prefix = f'Log Likelihood Evaluation :', suffix = 'Complete', length = 50)
			
			avg_ll.append(np.mean(batch))
			ll_dev.append(np.std(batch))
			


			test_data = [[np.nan]*len(self.params.feature_names)]
			m = meu(spmn, test_data)
			meus.append(m[0])


			
			
			env = get_env(self.dataset)
			total_reward = 0
			trials = 100000
			batch_size = trials / 10
			batch = list()

			for z in range(trials):
				
				state = env.reset()  #
				while(True):
					output = best_next_decision(spmn, state)
					#output = spmn_topdowntraversal_and_bestdecisions(spmn, test_data)
					action = output[0][0]
					state, reward, done = env.step(action)
					if done:
						total_reward += reward
						break
				if (z+1) % batch_size == 0:
					batch.append(total_reward/batch_size)
					total_reward = 0
				printProgressBar(z+1, trials, prefix = f'Average Reward Evaluation :', suffix = 'Complete', length = 50)

			avg_rewards.append(np.mean(batch))
			reward_dev.append(np.std(batch))
			
			
			
			print("\n\n\n\n\n")
			print(f"X-Means Limit: {limit}, \tVariables for splitting: {round(n)}")
			print("#Nodes: ",nodes[-1])
			print("Log Likelihood: ",avg_ll[-1])
			print("Log Likelihood Deviation: ",ll_dev[-1])
			print("MEU: ",meus[-1])
			print("Average rewards: ",avg_rewards[-1])
			print("Deviation: ",reward_dev[-1])
			print(nodes)
			print(meus)
			print("\n\n\n\n\n")
			
			
			plt.close()
			# plot line 
			plt.plot([original_stats[self.dataset]["ll"]]*len(avg_ll), linestyle="dotted", color ="red", label="LearnSPMN")
			plt.errorbar(np.arange(len(avg_ll)), avg_ll, yerr=ll_dev, marker="o", label="Anytime")
			plt.title(f"{self.dataset} Log Likelihood")
			plt.legend()
			if k is None:
				plt.savefig(f"{self.plot_path}/ll.png", dpi=100)
			else:
				plt.savefig(f"{self.plot_path}/{k}/ll.png", dpi=100)
			plt.close()
			
			plt.plot(meus, marker="o", label="Anytime")
			plt.plot([original_stats[self.dataset]["meu"]]*len(meus), linestyle="dotted", color ="red", label="LearnSPMN")
			plt.title(f"{self.dataset} MEU")
			plt.legend()
			if k is None:
				plt.savefig(f"{self.plot_path}/meu.png", dpi=100)
			else:
				plt.savefig(f"{self.plot_path}/{k}/meu.png", dpi=100)
			plt.close()

			plt.plot(nodes, marker="o", label="Anytime")
			plt.plot([original_stats[self.dataset]["nodes"]]*len(nodes), linestyle="dotted", color ="red", label="LearnSPMN")
			plt.title(f"{self.dataset} Nodes")
			plt.legend()
			if k is None:
				plt.savefig(f"{self.plot_path}/nodes.png", dpi=100)
			else:
				plt.savefig(f"{self.plot_path}/nodes.png", dpi=100)
			plt.close()

			original_reward = np.array([original_stats[self.dataset]["reward"]]*len(avg_rewards))
			dev = np.array([original_stats[self.dataset]["dev"]]*len(avg_rewards))
			plt.plot(original_reward, linestyle="dotted", color ="red", label="LearnSPMN")
			plt.fill_between(np.arange(len(avg_rewards)), original_reward-dev, original_reward+dev, alpha=0.3, color="red")
			plt.errorbar(np.arange(len(avg_rewards)), avg_rewards, yerr=reward_dev, marker="o", label="Anytime")
			plt.title(f"{self.dataset} Average Rewards")
			plt.legend()
			if k is None:
				plt.savefig(f"{self.plot_path}/rewards.png", dpi=100)
			else:
				plt.savefig(f"{self.plot_path}/rewards.png", dpi=100)
			plt.close()


			

			plt.plot(original_reward, linestyle="dotted", color ="red", label="LearnSPMN")
			plt.fill_between(np.arange(len(avg_rewards)), original_reward-dev, original_reward+dev, alpha=0.3, color="red")
			plt.errorbar(np.arange(len(avg_rewards)), avg_rewards, yerr=reward_dev, marker="o", label="Anytime")
			if original_reward[0] > 0:
				plt.axis(ymin=0, ymax=original_reward[0]*1.5)
			else:
				plt.axis(ymax=0, ymin=original_reward[0]*1.5)
			plt.title(f"{self.dataset} Average Rewards")
			plt.legend()
			if k is None:
				plt.savefig(f"{self.plot_path}/rewards_scaled1.png", dpi=100)
			else:
				plt.savefig(f"{self.plot_path}/rewards_scaled1.png", dpi=100)
			plt.close()

			
			lspmn_reward = str(abs(int(original_reward[0])))
			order = len(lspmn_reward)
			r_dev = np.array(reward_dev)
			if order > 1:
				 minl= (round(min(avg_rewards-r_dev)/(10**(order-2)) * 2)/2 - 0.5) * (10**(order-2))
				 maxl= (round(max(avg_rewards+r_dev)/(10**(order-2)) * 2)/2 + 0.5) * (10**(order-2))
			else:
				minl= round(min(avg_rewards-r_dev)*2)/2 - 0.5
				maxl= round(max(avg_rewards+r_dev)*2)/2 + 0.5
			plt.plot(original_reward, linestyle="dotted", color ="red", label="LearnSPMN")
			plt.fill_between(np.arange(len(avg_rewards)), original_reward-dev, original_reward+dev, alpha=0.3, color="red")
			plt.errorbar(np.arange(len(avg_rewards)), avg_rewards, yerr=reward_dev, marker="o", label="Anytime")
			plt.axis(ymin=minl, ymax=maxl)
			plt.title(f"{self.dataset} Average Rewards")
			plt.legend()
			if k is None:
				plt.savefig(f"{self.plot_path}/rewards_scaled2.png", dpi=100)
			else:
				plt.savefig(f"{self.plot_path}/rewards_scaled2.png", dpi=100)
			plt.close()

			

			
			f = open(f"{self.plot_path}/stats.txt", "w") if k is None else open(f"{self.plot_path}/{k}/stats.txt", "w")

			f.write(f"\n{self.dataset}")
			f.write(f"\n\tLog Likelihood : {avg_ll}")
			f.write(f"\n\tLog Likelihood Deviation: {ll_dev}")
			f.write(f"\n\tMEU : {meus}")
			f.write(f"\n\tNodes : {nodes}")
			f.write(f"\n\tAverage Rewards : {avg_rewards}")
			f.write(f"\n\tRewards Deviation : {reward_dev}")
			f.close()

			'''
			
			#except:
				#pass
			

			#past3 = avg_ll[-min(len(meus),3):]
				
			if n>=self.vars: #and round(np.std(past3), 3) <= 0.001:
				break


			i+=1
			limit += 1
			d+=1
			n = n+step

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



	def learn_aspmn_kfold(self, data, k):

		from sklearn.model_selection import KFold

		kfold = KFold(n_Splits=k, shuffle=True)
		cmap = plt.get_cmap('gnuplot')

		k_ = 1
		k_stats = dict()
		for trainidx, testidx in kfold.split(data):

			train, test = data[trainidx], data[testidx]
			_, stats = self.learn_aspmn(train, test, k=k_)
			k_stats[k] = stats
			k_+=1

		plt.close()
		maxlen = max([len(k_stats[i+1]["ll"]) for i in range(k)])
		total_ll = np.zeros(min([len(k_stats[i+1]["ll"]) for i in range(k)]))
		originalll = [original_stats[dataset]["ll"]] * maxlen
		plt.plot(originalll, linestyle="dotted", color ="blue", label="LearnSPN")
		for i in range(k):
			plt.plot(k_stats[i+1]["ll"], marker="o", color =cmap(i+1), label=(i+1))
			total_ll += np.array(k_stats[i+1]["ll"][:len(total_ll)])
		avg_ll = total_ll/k
		plt.plot(avg_ll, marker="o", color ="black", label="Mean")
		plt.title(f"{dataset} Log Likelihood")
		plt.legend()
		plt.savefig(f"{path}/{dataset}/ll.png", dpi=150)
		plt.close()

		maxlen = max([len(k_stats[i+1]["nodes"]) for i in range(k)])
		total_n = np.zeros(min([len(k_stats[i+1]["nodes"]) for i in range(k)]))
		originaln = [original_stats[dataset]["nodes"]] * maxlen
		plt.plot(originaln, linestyle="dotted", color ="blue", label="LearnSPN")
		for i in range(k):
			plt.plot(k_stats[i+1]["nodes"], marker="o", color =cmap(i+1), label=(i+1))
			total_n += np.array(k_stats[i+1]["nodes"][:len(total_n)])
		avg_n = total_n/k
		plt.plot(avg_n, marker="o", color ="black", label="Mean")
		plt.title(f"{dataset} Nodes")
		plt.legend()
		plt.savefig(f"{path}/{dataset}/nodes.png", dpi=150)
		plt.close()

		maxlen = max([len(k_stats[i+1]["meu"]) for i in range(k)])
		total_meu = np.zeros(min([len(k_stats[i+1]["meu"]) for i in range(k)]))
		originalmeu = [original_stats[dataset]["meu"]] * maxlen
		plt.plot(originalmeu, linestyle="dotted", color ="blue", label="LearnSPN")
		for i in range(k):
			plt.plot(k_stats[i+1]["meu"], marker="o", color =cmap(i+1), label=(i+1))
			total_meu += np.array(k_stats[i+1]["meu"][:len(total_meu)])
		avg_meu = total_meu/k
		plt.plot(avg_meu, marker="o", color ="black", label="Mean")
		plt.title(f"{dataset} MEU")
		plt.legend()
		plt.savefig(f"{path}/{dataset}/meu.png", dpi=150)
		plt.close()

		maxlen = max([len(k_stats[i+1]["reward"]) for i in range(k)])
		total_r = np.zeros(min([len(k_stats[i+1]["reward"]) for i in range(k)]))
		originalr = [original_stats[dataset]["reward"]] * maxlen
		plt.plot(originalr, linestyle="dotted", color ="blue", label="LearnSPN")
		for i in range(k):
			plt.errorbar(np.arange(len(k_stats[i+1]["reward"])), k_stats[i+1]["reward"], yerr=k_stats[i+1]["deviation"], marker="o", color =cmap(i+1), label=(i+1))
			total_r += np.array(k_stats[i+1]["reward"][:len(total_r)])
		avg_r = total_r/k
		plt.plot(avg_r, marker="o", color ="black", label="Mean")
		plt.title(f"{dataset} Average Rewards")
		plt.legend()
		plt.savefig(f"{path}/{dataset}/reward.png", dpi=150)
		plt.close()




class SPMNParams:

	def __init__(self, partial_order, decision_nodes, utility_nodes, feature_names, feature_labels, meta_types, util_to_bin):

		self.partial_order = partial_order
		self.decision_nodes = decision_nodes
		self.utility_nodes = utility_nodes
		self.feature_names = feature_names
		self.feature_labels = feature_labels
		self.meta_types = meta_types
		self.util_to_bin = util_to_bin
