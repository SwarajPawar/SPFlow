
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


# Anytime SPMN class
class Anytime_SPMN:

	def __init__(self, dataset, output_path, partial_order, decision_nodes, utility_node, feature_names, feature_labels,
			meta_types, cluster_by_curr_information_set=False, util_to_bin=False):

		#Save the parameters
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

		#Create output directory if it doesn't exist
		self.plot_path = f"{output_path}/{dataset}"
		if not pth.exists(self.plot_path):
			try:
				os.makedirs(self.plot_path)
			except OSError:
				print ("Creation of the directory %s failed" % self.plot_path)
				sys.exit()


	#Get and set operations
	def set_next_operation(self, next_op):
		self.op = next_op

	def get_curr_operation(self):
		return self.op

	#Function to learn the SPMN
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

	#Return log-likelihood for the given test instance
	def get_loglikelihood(self, instance):
        test_data = np.array(instance).reshape(-1, len(self.params.feature_names))
        return log_likelihood(self.spmn, test_data)[0][0]


    # Get reward by simulating policy in the environment
    def get_reward(self, id):

        state = self.env.reset()
        while(True):
            output = best_next_decision(self.spmn, state)
            action = output[0][0]
            state, reward, done = self.env.step(action)
            if done:
                return reward

    #Get policy by running spmn in environment
    def get_policy(self, ids):

        policy = ""
        state = self.env.reset()
        while(True):
            output = best_next_decision(self.spmn, state)
            action = output[0][0]
            policy += f"{action}  "
            state, reward, done = self.env.step(action)
            if done:
                return policy

	def learn_aspmn(self, train, test=None, get_stats = False, save_models=True):
		"""
		:param: train dataset
		:return: learned ASPMNs
		"""
		
		
		if save_models:
			#Create output directory if it doesn't exist
			if not pth.exists(f'{self.plot_path}/models'):
				try:
					os.makedirs(f'{self.plot_path}/models')
				except OSError:
					print ("Creation of the directory %s failed" % f'{self.plot_path}/models')
					sys.exit()

        
        #Initial parameters
        limit = 2 
        n = int(self.vars**0.5)
        step = 0 if self.vars < 10 else (self.vars - (self.vars**0.5) + 1)/10
        d = 2

        #Initialize lists for storing statistics over iterations
		all_avg_ll = list()
		all_ll_dev = list()
		all_meus = list()
		all_nodes = list()
		all_avg_rewards = list()
		all_reward_dev = list()


        #Start Anytime iterations
		i = 0
		while(True):

			index = 0
			print(f"\nIteration: {i}\n")
			
			#Get Current and remaining scopes and initialize next operation
			curr_information_set_scope = np.array(range(len(self.params.partial_order[0]))).tolist()
			remaining_vars_scope = np.array(range(len(self.params.feature_names))).tolist()
			self.set_next_operation('Any')
			self.limit = limit 
			self.n = n  
			self.d = d

			#Start Learning the network
			print("\nStart Learning...")
			spmn = self.__learn_spmn_structure(train, remaining_vars_scope, curr_information_set_scope, index)
			print("SPMN Learned")
			#spmn = Prune(spmn)
			self.spmn = spmn

			stats = None


			if get_stats:
				#Store the stats in a dictionary
				avg_ll, ll_dev = self.evaluate_loglikelihood(test)
				meu_ = self.evaluate_meu()
				nodes = self.evaluate_nodes()
				avg_rewards, reward_dev = self.evaluate_rewards()

				all_avg_ll.append(avg_ll)
				all_ll_dev.append(ll_dev)
				all_meus.append(meu_)
				all_nodes.append(nodes)
				all_avg_rewards.append(avg_rewards)
				all_reward_dev.append(reward_dev)

		        stats = {"ll" : all_avg_ll,
		                "ll_dev": all_ll_dev,
		                "meu" : all_meus,
		                "nodes" : all_nodes,
		                "reward" : all_avg_rewards,
		                "reward_dev" : all_reward_dev
		                }
            
	            #Print the stats
				print("\n\n\n\n\n")
				print(f"X-Means Limit: {limit}, \tVariables for splitting: {round(n)}")
				print("#Nodes: ",nodes)
				print("Log Likelihood: ",avg_ll)
				print("Log Likelihood Deviation: ",ll_dev)
				print("MEU: ",meu_)
				print("Average rewards: ",avg_rewards)
				print("Deviation: ",reward_dev)
				print("\n\n\n\n\n")

				#Save the stats in a file
				f = open(f"{self.plot_path}/stats.txt", "w")

				f.write(f"\n{self.dataset}")
				f.write(f"\n\tLog Likelihood : {avg_ll}")
				f.write(f"\n\tLog Likelihood Deviation: {ll_dev}")
				f.write(f"\n\tMEU : {meus}")
				f.write(f"\n\tNodes : {nodes}")
				f.write(f"\n\tAverage Rewards : {avg_rewards}")
				f.write(f"\n\tRewards Deviation : {reward_dev}")
				f.close()
			
			
	        
	        #Return the network and stats for the current iteration
	        yield self.spmn, stats
			
			
			#Termination criterion
			if n>self.vars:  #and round(np.std(past3), 3) <= 0.001:
                break

            #Update the parameter values
            i += 1
            limit += 1
            d += 1
            n = n+step
            if self.vars < 10:
                step = 1

        

    def evaluate_nodes(self, spmn=self.spmn):
    	#Get nodes in the network
		return get_structure_stats_dict(spmn)["nodes"]
		
	def evaluate_loglikelihood_parallel(self, test, spmn=self.spmn, batches=10):

		if not test:
			return None, None

		#Initilize parameters for Log-likelihood evaluation
		total_ll = 0
        trials1 = test.shape[0]
        batch_size = int(trials1 / batches)
        batch = list()
        pool = multiprocessing.Pool()

        
        #Get average log-likelihood for the batches
        for b in range(batches):
            test_slice = test[b*batch_size:(b+1)*batch_size]
            lls = pool.map(self.get_loglikelihood, test_slice)
            total_ll = sum(lls)
            batch.append(total_ll/batch_size)
            printProgressBar(b+1, 10, prefix = f'Log Likelihood Evaluation :', suffix = 'Complete', length = 50)
        
        #Get average ll and deviation
        avg_ll = np.mean(batch)
        ll_dev = np.std(batch)

        return avg_ll, ll_dev

    def evaluate_loglikelihood_sequential(self, test, spmn=self.spmn, batches=10):

        if not test:
            return None, None

        #Initilize parameters for Log-likelihood evaluation
        total_ll = 0
        trials1 = test.shape[0]
        batch_size = int(trials1 / batches)
        batch = list()

        
        #Get average log-likelihood for the batches
        for b in range(batches):
            test_slice = test[b*batch_size:(b+1)*batch_size]
            lls = list()
            for instance in test_slice:
                lls.append(self.get_loglikelihood(instance))
                printProgressBar(b+1, batches, prefix = f'Log Likelihood Evaluation :', suffix = 'Complete', length = 50)
            total_ll = sum(lls)
            batch.append(total_ll/batch_size)
            
        
        #Get average ll and deviation
        avg_ll = np.mean(batch)
        ll_dev = np.std(batch)

        return avg_ll, ll_dev
		

    def evaluate_meu(self, spmn=self.spmn):
        #Compute the MEU of the Network
		test_data = [[np.nan]*len(self.params.feature_names)]
		m = meu(spmn, test_data) 
		return m[0]


	def evaluate_rewards_parallel(self, spmn=self.spmn, batch_size = 20000, batches = 25):

		#Initialize domain environment
		self.env = get_env(self.dataset)

		if not self.env:
			return None, None

		#Initialize parameters for computing rewards
		total_reward = 0
        reward_batch = list()
        
        pool = multiprocessing.Pool()
        #Get the rewards parallely for each batch
        for y in range(batches):
            ids = [None for x in range(batch_size)]
            rewards = pool.map(self.get_reward, ids)
            reward_batch.append(sum(rewards) / batch_size)
            printProgressBar(y+1, batches, prefix = f'Average Reward Evaluation :', suffix = 'Complete', length = 50)

        #get the mean and std dev of the rewards    
        avg_rewards = np.mean(reward_batch)
        reward_dev = np.std(reward_batch)

        return avg_rewards, reward_dev

    def evaluate_rewards_sequential(self, spmn=self.spmn, batch_size = 20000, batches = 25):

        #Initialize domain environment
        self.env = get_env(self.dataset)

        if not self.env:
            return None, None

        #Initialize parameters for computing rewards
        total_reward = 0
        reward_batch = list()

        #Get the rewards parallely for each batch
        for y in range(batches):
            rewards = list()
            for z in range(batch_size):
                rewards.append(self.get_reward(z))
                printProgressBar(y*batch_size + z+1, batches*batch_size, prefix = f'Average Reward Evaluation :', suffix = 'Complete', length = 50)
            reward_batch.append(sum(rewards) / batch_size)
            

        #get the mean and std dev of the rewards    
        avg_rewards = np.mean(reward_batch)
        reward_dev = np.std(reward_batch)

        return avg_rewards, reward_dev

#Object to store SPMN parameters
class SPMNParams:

	def __init__(self, partial_order, decision_nodes, utility_nodes, feature_names, feature_labels, meta_types, util_to_bin):

		self.partial_order = partial_order
		self.decision_nodes = decision_nodes
		self.utility_nodes = utility_nodes
		self.feature_names = feature_names
		self.feature_labels = feature_labels
		self.meta_types = meta_types
		self.util_to_bin = util_to_bin
