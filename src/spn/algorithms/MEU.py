"""
Created on March 28, 2019
@author: Hari Teja Tatavarti

"""

from spn.algorithms.MPE import get_node_funtions
from spn.algorithms.Inference import  likelihood, max_likelihood, log_likelihood, sum_likelihood, prod_likelihood
from spn.algorithms.Validity import is_valid
from spn.structure.Base import get_nodes_by_type, Max, Leaf, Sum, Product, get_topological_order_layers
from spn.structure.leaves.histogram.Inference import histogram_likelihood
from spn.structure.leaves.spmnLeaves.SPMNLeaf import Utility
import numpy as np


from collections import defaultdict
import collections


def merge_input_vals(l):
    return np.concatenate(l)

def meu_sum(node, meu_per_node, data=None, lls_per_node=None, rand_gen=None):
    meu_children = meu_per_node[:,[child.id for child in node.children]]
    likelihood_children = lls_per_node[:,[child.id for child in node.children]]
    weighted_likelihood = np.array(node.weights)*likelihood_children
    norm = np.sum(weighted_likelihood, axis=1)
    normalized_weighted_likelihood = weighted_likelihood / norm.reshape(-1,1)
    meu_per_node[:,node.id] = np.sum(meu_children * normalized_weighted_likelihood, axis=1)

def meu_prod(node, meu_per_node, data=None, lls_per_node=None, rand_gen=None):
    # product node just passes up the utils of whichever child contains util nodes
    meu_children = meu_per_node[:,[child.id for child in node.children]]
    for meu_child in np.isnan(meu_children[0]):
        if not meu_child:
            meu_per_node[:,node.id] = meu_children[~np.isnan(meu_children)]
            return
    meu_per_node[:,node.id] = np.nan
    # the line below works because product nodes should have only one child containing the utility node.
    # if more than one utility column is allowed this will have to change.
    

def meu_max(node, meu_per_node, data=None, lls_per_node=None, rand_gen=None):
    meu_children = meu_per_node[:, [child.id for child in node.children]]
    decision_value_given = data[:, node.dec_idx]
    max_value = np.argmax(meu_children, axis=1)
    # if data contains a decision value use that otherwise use max
    child_idx = np.select([np.isnan(decision_value_given), True],
                          [max_value, decision_value_given]).astype(int)
    child_idx_to_id = lambda idx: node.children[idx].id
    child_idx_to_id = np.vectorize(child_idx_to_id)
    child_id = child_idx_to_id(child_idx)
    meu_per_node[:,node.id] = meu_per_node[np.arange(meu_per_node.shape[0]),child_id]

def meu_util(node, meu_per_node, data=None, lls_per_node=None, rand_gen=None):
    #returns average value of the utility node
    util_value = 0
    for i in range(len(node.bin_repr_points)):
        util_value += node.bin_repr_points[i] * node.densities[i]
    utils = np.empty((data.shape[0]))
    utils[:] = util_value
    meu_per_node[:,node.id] = utils * lls_per_node[:,node.id]


_node_bottom_up_meu = {Sum: meu_sum, Product: meu_prod, Max: meu_max, Utility: meu_util}

def meu(node, input_data,
        node_bottom_up_meu=_node_bottom_up_meu,
        in_place=False):
    # valid, err = is_valid(node)
    # assert valid, err
    if in_place:
        data = input_data
    else:
        data = np.array(input_data)
    # assumes utility is only one and is at the last
    # print("input data:", input_data[:, -1])
    assert np.isnan(data[:, -1]), "Please specify utility variable as NaN"
    nodes = get_nodes_by_type(node)
    likelihood_per_node = np.zeros((data.shape[0], len(nodes)))
    meu_per_node = np.zeros((data.shape[0], len(nodes)))
    meu_per_node.fill(np.nan)
    # one pass bottom up evaluating the likelihoods
    likelihood(node, data, dtype=data.dtype, lls_matrix=likelihood_per_node)
    eval_spmn_bottom_up_meu(
            node,
            _node_bottom_up_meu,
            meu_per_node=meu_per_node,
            data=data,
            lls_per_node=likelihood_per_node
        )
    result = meu_per_node[:,node.id]
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
