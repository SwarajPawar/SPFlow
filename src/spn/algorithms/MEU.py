"""
Created on March 28, 2019
@author: Hari Teja Tatavarti

"""

from spn.algorithms.MPE import mpe, get_node_funtions
from spn.algorithms.Inference import  likelihood, max_likelihood, log_likelihood, sum_likelihood, prod_likelihood
from spn.algorithms.Validity import is_valid
from spn.structure.Base import assign_ids, get_nodes_by_type, Max, Leaf, Sum, Product, get_topological_order_layers
from spn.structure.leaves.histogram.Inference import histogram_likelihood
from spn.structure.leaves.spmnLeaves.SPMNLeaf import State, Utility
import numpy as np
from copy import deepcopy


def meu_sum(node, meu_per_node, data=None, lls_per_node=None, rand_gen=None):
    meu_children = meu_per_node[:,[child.id for child in node.children]]
    likelihood_children = lls_per_node[:,[child.id for child in node.children]]
    weighted_likelihood = np.array(node.weights)*likelihood_children
    weighted_likelihood[weighted_likelihood < 1/(10**10)] = 0
    norm = np.sum(weighted_likelihood, axis=1)
    if norm > 1/(10**10):
        normalized_weighted_likelihood = weighted_likelihood / norm.reshape(-1,1)
        meu_per_node[:,node.id] = np.sum(meu_children * normalized_weighted_likelihood, axis=1)
    else:
        meu_per_node[:,node.id] = 0

def meu_prod(node, meu_per_node, data=None, lls_per_node=None, rand_gen=None):
    # product node adds together the utilities of its children
    # if there is only one utility node then only one child of each product node
    # will have a utility value
    meu_children = meu_per_node[:,[child.id for child in node.children]]
    meu_per_node[:,node.id] = np.nansum(meu_children,axis=1)

def meu_max(node, meu_per_node, data=None, lls_per_node=None, rand_gen=None):
    meu_children = meu_per_node[:, [child.id for child in node.children]]
    decision_value_given = data[:, node.dec_idx]
    max_value = np.argmax(meu_children, axis=1)
    d_given = np.full(decision_value_given.shape[0], np.nan)
    mapd = {node.dec_values[i]:i for i in range(len(node.dec_values))}
    for k, v in mapd.items(): d_given[decision_value_given==k] = v
    # if data contains a decision value use that otherwise use max
    child_id = np.select([np.isnan(d_given), True],
                          [max_value, d_given]).astype(int)
    meu_node = meu_children[np.arange(meu_children.shape[0]),child_id]
    # if decision value given is not in children, assign 0 utility
    missing_dec_branch = np.logical_and(np.logical_not(np.isnan(decision_value_given)),np.isnan(d_given))
    meu_node[missing_dec_branch] = 0
    meu_per_node[:,node.id] = meu_node

def meu_util(node, meu_per_node, data=None, lls_per_node=None, rand_gen=None):
    #returns average value of the utility node
    util_value = 0.0
    for i in range(len(node.bin_repr_points)):
        util_value += node.bin_repr_points[i] * node.densities[i]
    util_value /= sum(node.densities)
    utils = np.empty((data.shape[0]))
    utils[:] = util_value
    meu_per_node[:,node.id] = utils * lls_per_node[:,node.id]


_node_bottom_up_meu = {Sum: meu_sum, Product: meu_prod, Max: meu_max, Utility: meu_util}

def meu(root, input_data,
        node_bottom_up_meu=_node_bottom_up_meu,
        in_place=False):
    # valid, err = is_valid(node)
    # assert valid, err
    if in_place:
        data = input_data
    else:
        data = np.copy(input_data)
    nodes = get_nodes_by_type(root)
    utility_scope = set()
    for node in nodes:
        if type(node) is Utility:
            utility_scope.add(node.scope[0])
    assert np.all(np.isnan(data[np.arange(data.shape[0]), list(utility_scope)])), "Please specify all utility values as np.nan"
    likelihood_per_node = np.zeros((data.shape[0], len(nodes)))
    meu_per_node = np.zeros((data.shape[0], len(nodes)))
    meu_per_node.fill(np.nan)
    # one pass bottom up evaluating the likelihoods
    likelihood(root, data, dtype=data.dtype, lls_matrix=likelihood_per_node)
    eval_spmn_bottom_up_meu(
            root,
            _node_bottom_up_meu,
            meu_per_node=meu_per_node,
            data=data,
            lls_per_node=likelihood_per_node
        )
    result = meu_per_node[:,root.id]
    return result


def eval_spmn_bottom_up_meu(root, eval_functions, meu_per_node=None, data=None, lls_per_node=None):
    """
      evaluates an spn top to down
      :param root: spnt root
      :param eval_functions: is a dictionary that contains k:Class of the node, v:lambda function that receives as parameters (node, [parent_results], args**) and returns {child : intermediate_result}. This intermediate_result will be passed to child as parent_result. If intermediate_result is None, no further propagation occurs
      :param all_results: is a dictionary that contains k:Class of the node, v:result of the evaluation of the lambda function for that node.
      :param parent_result: initial input to the root node
      :param args: free parameters that will be fed to the lambda functions.
      :return: the result of computing and propagating all the values throught the network, decisions at each max node for the instances reaching that max node.
      """
    for node_type, func in eval_functions.items():
        if "_eval_func" not in node_type.__dict__:
            node_type._eval_func = []
        node_type._eval_func.append(func)
    for layer in get_topological_order_layers(root):
        for n in layer:
            if type(n)==Max or type(n)==Sum or type(n)==Product or type(n)==Utility:
                func = n.__class__._eval_func[-1]
                func(n, meu_per_node, data=data, lls_per_node=lls_per_node)
    for node_type, func in eval_functions.items():
        del node_type._eval_func[-1]
        if len(node_type._eval_func) == 0:
            delattr(node_type, "_eval_func")


def eval_spmn_top_down_meu(root, eval_functions,
        all_results=None, parent_result=None, data=None,
        lls_per_node=None, likelihood_per_node=None):
    """
      evaluates an spn top to down
      :param root: spnt root
      :param eval_functions: is a dictionary that contains k:Class of the node, v:lambda function that receives as parameters (node, [parent_results], args**) and returns {child : intermediate_result}. This intermediate_result will be passed to child as parent_result. If intermediate_result is None, no further propagation occurs
      :param all_results: is a dictionary that contains k:Class of the node, v:result of the evaluation of the lambda function for that node.
      :param parent_result: initial input to the root node
      :param args: free parameters that will be fed to the lambda functions.
      :return: the result of computing and propagating all the values throught the network, decisions at each max node for the instances reaching that max node.
      """
    if all_results is None:
        all_results = {}
    else:
        all_results.clear()
    all_decisions = []
    all_max_nodes = []
    for node_type, func in eval_functions.items():
        if "_eval_func" not in node_type.__dict__:
            node_type._eval_func = []
        node_type._eval_func.append(func)
    all_results[root] = [parent_result]
    for layer in reversed(get_topological_order_layers(root)):
        for n in layer:
            func = n.__class__._eval_func[-1]
            param = all_results[n]
            if type(n) == Max:
                result, decision_values, max_nodes = func(n, param,
                                    data=data, lls_per_node=lls_per_node)
                all_decisions.append(decision_values)
                all_max_nodes.append(max_nodes)
            elif type(n) == Sum:
                result = func(n, param,
                        likelihood_per_node=likelihood_per_node,
                        data=data,
                        lls_per_node=lls_per_node
                    )
            else:
                result = func(n, param, data=data, lls_per_node=lls_per_node)
            if result is not None and not isinstance(n, Leaf):
                assert isinstance(result, dict)
                for child, param in result.items():
                    if child not in all_results:
                        all_results[child] = []
                    all_results[child].append(param)
    for node_type, func in eval_functions.items():
        del node_type._eval_func[-1]
        if len(node_type._eval_func) == 0:
            delattr(node_type, "_eval_func")
    return all_results[root], all_decisions, all_max_nodes

def best_next_decision(spmn, input_data, depth=1, in_place=False):
    root = spmn.spmn_structure
    if in_place:
        data = input_data
    else:
        data = np.copy(input_data)
    nodes = get_nodes_by_type(root)
    dec_dict = {}
    # find all possible decision values
    for node in nodes:
        if type(node) == Max:
            if node.dec_idx in dec_dict:
                dec_dict[node.dec_idx].union(set(node.dec_values))
            else:
                dec_dict[node.dec_idx] = set(node.dec_values)
    next_dec_idx = None
    # find next undefined decision
    for idx in dec_dict.keys():
        if np.all(np.isnan(data[:,idx])):
            next_dec_idx = idx
            break
    assert next_dec_idx != None, "please assign all values of next decision to np.nan"
    # determine best decisions based on meu
    dec_vals = list(dec_dict[next_dec_idx])
    best_decisions = np.full((1,data.shape[0]),dec_vals[0])
    data[:,next_dec_idx] = best_decisions
    if depth == 1:
        meu_best = meu(root, data)
    else:
        meu_best = np.array([rmeu(spmn, data[0], depth)])
    for i in range(1, len(dec_vals)):
        decisions_i = np.full((1,data.shape[0]), dec_vals[i])
        data[:,next_dec_idx] = decisions_i
        if depth == 1:
            meu_i = meu(root, data)
        else:
            meu_i = np.array([rmeu(spmn, data[0], depth)])
        best_decisions = np.select([np.greater(meu_i, meu_best),True],[decisions_i, best_decisions])
        data[:,next_dec_idx] = best_decisions
        meu_best = np.maximum(meu_i,meu_best)
    return best_decisions

def rmeu(rspmn, input_data, depth=2): # maybe TODO add args for epsilon (e-greedy exploration) and horizon discount
    assert type(depth) is int and depth > 0, "depth must be a positive integer."
    rspmn_root = rspmn.spmn_structure
    # find indices of s1_and_decisions
    nodes = get_nodes_by_type(rspmn_root)
    dec_indices = [] # TODO determine based on feature_names and decision_nodes
    for node in nodes:
        if isinstance(node, Max):
            dec_idx = node.dec_idx
            if dec_idx not in dec_indices:
                dec_indices.append(dec_idx)
    dec_indices.sort()
    # caching unconditioned MEUs
    s1_and_decisions_to_s2, s1_and_depth_to_meu = build_rspmn_meu_caches(
                                                    rspmn,
                                                    dec_indices,
                                                    depth
                                                )
    # TODO clean up by combining and caching more of the following steps
    # find state_branch corresponding to the s1 value in input
    # if no state is specified, we must use overall model
    input_state_branch = rspmn_root
    input_s1_node = None
    if not np.isnan(input_data[0]):
        for state_branch in rspmn_root.children:
            branch_s1 = state_branch.children[0]
            if branch_s1.densities[int(input_data[0])] > 0.000000001:
                input_state_branch = state_branch
                break
        input_s1_node = input_state_branch.children[0]
    # find all possible s1_and_decisions paths given input_data
    specified_s1_and_decisions = [input_s1_node]+[input_data[dec_idx] for dec_idx in dec_indices]
    s1_and_dec_paths = list()
    for s1_and_decisions in s1_and_decisions_to_s2.keys():
        match = True
        # if s1 is specified then make sure it matches
        if not specified_s1_and_decisions[0] is None and\
                s1_and_decisions[0] != specified_s1_and_decisions[0]:
            match = False
        # make sure all specified decisions match
        for i in range(1, len(s1_and_decisions)):
            if not np.isnan(specified_s1_and_decisions[i]) and\
                    s1_and_decisions[i] != specified_s1_and_decisions[i]:
                match = False
        # if the specified components match then add to
        if match: s1_and_dec_paths.append(s1_and_decisions)
    result_meu = None
    for path in s1_and_dec_paths:
        path_s2_nodes = s1_and_decisions_to_s2[path]
        decisions = path[1:]
        s1_and_dec_data = deepcopy(input_data)
        path_meu = 0
        for i in range(len(dec_indices)):
            s1_and_dec_data[dec_indices[i]] = decisions[i]
        s2_prob_norm = 0
        for s2_node in path_s2_nodes:
            s2_val = np.argmax(s2_node.densities)
            # add s2 value to input vector
            s1_and_dec_data[-1] = s2_val
            # TODO calculate these from the s1 branch rather than from rspmn_root
            s2_prob = likelihood(rspmn_root, np.array([s1_and_dec_data])) # TODO cache these ((?? maybe too niche bc variables other than s1 and decs influence as well))
            s2_meu = meu(rspmn_root, np.array([s1_and_dec_data])) # TODO should have already been cached for path
            s2_util = s2_meu
            # total s2 value is value at current depth + value of its corresponding s1 one step deeper
            if depth > 1 and len(s2_node.interface_links)>0:
                s2_count = sum([count for count in s2_node.interface_links.values()])
                for linked_s1, count in s2_node.interface_links.items():
                    s2_util += s1_and_depth_to_meu[(linked_s1,depth-1)] * (count/s2_count)
            else:
                s2_util = s2_meu
            s2_prob_norm += s2_prob
            path_meu += s2_prob * s2_util
        # normalize summed meu w.r.t. s2 probabilities
        path_meu /= s2_prob_norm
        if result_meu is None or path_meu > result_meu:
            result_meu = path_meu
    return result_meu


def build_rspmn_meu_caches(rspmn, dec_indices, depth=2):
    rspmn_root = rspmn.spmn_structure
    scope_len = rspmn_root.scope[-1]
    # collecting or creating caches
    if hasattr(rspmn,"s1_and_decisions_to_s2") and rspmn.s1_and_decisions_to_s2:
        s1_and_decisions_to_s2 = rspmn.s1_and_decisions_to_s2
    else:
        s1_and_decisions_to_s2 = get_s1_and_decisions_to_s2(rspmn_root)
    if hasattr(rspmn, "s1_to_branch") and rspmn.s1_to_branch:
        s1_to_branch = rspmn.s1_to_branch
        s1_nodes = [state_branch.children[0] for state_branch in rspmn_root.children]
    else:
        s1_to_branch = dict()
        s1_nodes = list()
        for state_branch in rspmn_root.children:
            s1_node = state_branch.children[0]
            s1_nodes.append(s1_node)
            s1_to_branch[s1_node] = state_branch
    if hasattr(rspmn,"s1_and_depth_to_meu") and rspmn.s1_and_depth_to_meu:
        s1_and_depth_to_meu = rspmn.s1_and_depth_to_meu
    else:
        s1_and_depth_to_meu = dict()
        for s1_node in s1_nodes:
            state_branch = s1_to_branch[s1_node]
            state_branch = assign_ids(state_branch)
            #s1_val = np.argmax(s1_node.densities)
            meu_data_s1 = np.array([[np.nan]+[np.nan]*(scope_len)])
            s1_and_depth_to_meu[(s1_node, 1)] = meu(state_branch,meu_data_s1).reshape(-1)
    if hasattr(rspmn, "s1_and_dec_to_s2_prob") and rspmn.s1_and_dec_to_s2_prob:
        s1_and_dec_to_s2_prob = rspmn.s1_and_dec_to_s2_prob
    else:
        s1_and_dec_to_s2_prob = dict()
    if hasattr(rspmn, "s2_to_meu") and rspmn.s2_to_meu:
        s2_to_meu = rspmn.s2_to_meu
    else:
        s2_to_meu = dict()
    max_cached_depth = max([s1_d[1] for s1_d in s1_and_depth_to_meu.keys()])
    for d in range(max_cached_depth+1,depth+1):
        for s1_and_decisions, s2_nodes in s1_and_decisions_to_s2.items():
            # create input vector for likelihood and MEU calculations
            s1_node = s1_and_decisions[0]
            state_branch = s1_to_branch[s1_node]
            s1_val = np.argmax(s1_node.densities)
            decisions = s1_and_decisions[1:]
            s1_and_dec_data = [s1_val]+[np.nan]*(scope_len)
            for i in range(len(dec_indices)):
                s1_and_dec_data[dec_indices[i]] = decisions[i]
            s2_prob_norm = 0
            s1_and_decisions_meu = 0
            for s2_node in s2_nodes:
                s2_val = np.argmax(s2_node.densities)
                # add s2 value to input vector
                s1_and_dec_data[-1] = s2_val
                path_data = np.array(s1_and_dec_data)
                state_branch = assign_ids(state_branch)
                if tuple(path_data) in s1_and_dec_to_s2_prob:
                    s2_prob = s1_and_dec_to_s2_prob[tuple(path_data)]
                else:
                    s2_prob = likelihood(state_branch,path_data.reshape(1,-1)).reshape(-1)
                    s1_and_dec_to_s2_prob[tuple(path_data)] = s2_prob
                if s2_node in s2_to_meu:
                    s2_meu = s2_to_meu[s2_node]
                else:
                    s2_meu = meu(state_branch, np.array([s1_and_dec_data])).reshape(-1)
                    s2_to_meu[s2_node] = deepcopy(s2_meu)
                # total s2 value is value at current depth + value of its corresponding s1 one step deeper
                s2_util = deepcopy(s2_meu)
                s2_count = sum([count for count in s2_node.interface_links.values()])
                for linked_s1, count in s2_node.interface_links.items():
                    if (linked_s1,d-1) in s1_and_depth_to_meu:
                        s2_util += s1_and_depth_to_meu[(linked_s1,d-1)] * (count/s2_count)
                s2_prob_norm += s2_prob
                s1_and_decisions_meu += s2_prob * s2_util
            # normalize summed meu w.r.t. s2 probabilities
            s1_and_decisions_meu /= s2_prob_norm
            # s1 meu at current depth is meu of best decision path
            if (s1_node, d) in s1_and_depth_to_meu:
                s1_and_depth_to_meu[(s1_node, d)] = max(
                        s1_and_decisions_meu,
                        s1_and_depth_to_meu[(s1_node, d)]
                    )
            else:
                s1_and_depth_to_meu[(s1_node, d)] = s1_and_decisions_meu
    # storing caches
    setattr(rspmn,"s1_and_decisions_to_s2",s1_and_decisions_to_s2)
    setattr(rspmn, "s1_to_branch", s1_to_branch)
    setattr(rspmn,"s1_and_depth_to_meu",s1_and_depth_to_meu)
    setattr(rspmn, "s1_and_dec_to_s2_prob",s1_and_dec_to_s2_prob)
    setattr(rspmn, "s2_to_meu",s2_to_meu)
    rspmn_root = assign_ids(rspmn_root)
    return s1_and_decisions_to_s2, s1_and_depth_to_meu

def get_s1_and_decisions_to_s2(rspmn_root):
    s1_and_decisions_to_s2 = dict()
    for state_branch in rspmn_root.children:
        s1 = state_branch.children[0]
        queue = state_branch.children[1:]
        fill_s1_and_decisions_to_s2(s1_and_decisions_to_s2, queue, [s1])
    return s1_and_decisions_to_s2

def fill_s1_and_decisions_to_s2(s1_and_decisions_to_s2, queue, path):
    while len(queue) > 0:
        node = queue.pop(0)
        if isinstance(node, Max):
            for i in range(len(node.dec_values)):
                dec_val_i = node.dec_values[i]
                child_i = node.children[i]
                fill_s1_and_decisions_to_s2(
                        s1_and_decisions_to_s2,
                        [child_i],
                        path+[dec_val_i]
                    )
        elif isinstance(node, State):
            if tuple(path) in s1_and_decisions_to_s2:
                s1_and_decisions_to_s2[tuple(path)] += [node]
            else:
                s1_and_decisions_to_s2[tuple(path)] = [node]
        elif isinstance(node, Product) or isinstance(node, Sum):
            for child in node.children:
                queue.append(child)
