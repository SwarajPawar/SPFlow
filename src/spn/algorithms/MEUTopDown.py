from spn.algorithms.MEU import meu
from spn.algorithms.MPE import get_node_funtions
from spn.algorithms.Inference import  likelihood, max_likelihood, log_likelihood, sum_likelihood, prod_likelihood
from spn.algorithms.Validity import is_valid
from spn.structure.Base import get_nodes_by_type, Max, Leaf, Sum, Product, get_topological_order_layers
from spn.structure.leaves.histogram.Inference import histogram_likelihood
from spn.structure.leaves.spmnLeaves.SPMNLeaf import Utility
import numpy as np

from spn.algorithms.Inference import interface_switch_log_likelihood
from spn.structure.Base import InterfaceSwitch

def merge_input_vals(l):
    return np.concatenate(l)

def max_best_dec_with_meu(node, parent_result, data=None, meu_per_node=None, rand_gen=None):
    if len(parent_result) == 0:
        return None

    parent_result = merge_input_vals(parent_result)

    # assert data is not None, "data must be passed through to max nodes for proper evaluation."
    decision_value_given = data[parent_result, node.dec_idx]

    w_children_meu = np.zeros((len(parent_result), len(node.dec_values)))
    for i, c in enumerate(node.children):
        w_children_meu[:, i] = meu_per_node[parent_result, c.id]
        decision_value_given[decision_value_given == node.dec_values[i]] = i

    max_value = np.argmax(w_children_meu, axis=1)
    #print(f'w_children_meu {w_children_meu}')
    # if data contains a decision value use that otherwise use max
    max_child_branches = np.select([np.isnan(decision_value_given), True],
                                   [max_value, decision_value_given]).astype(int)

    children_row_ids = {}

    # Decisions given are not in the set of decisions the node holds.
    # Leave as is and not pass values to children

    for i, c in enumerate(node.children):
        children_row_ids[c] = parent_result[max_child_branches == i]
        data[children_row_ids[c], node.dec_idx] = \
            node.dec_values[max_child_branches]

    return children_row_ids


def eval_spmn_top_down(root, eval_functions,
        all_results=None, parent_result=None, data=None,
        lls_per_node=None, meu_per_node=None):
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
                result = func(n, param, data=data, meu_per_node=meu_per_node)
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

    return all_results[root]

def spmn_topdowntraversal_and_bestdecisions(
    node,
    input_data,
    in_place=False,
):
    valid, err = is_valid(node)
    assert valid, err

    assert np.all(
        np.any(np.isnan(input_data), axis=1)
    ), "each row must have at least a nan value where the samples will be substituted"

    if in_place:
        data = input_data
    else:
        data = np.array(input_data)

    node_functions = get_node_funtions()
    _node_functions_top_down = node_functions[0].copy()
    _node_functions_top_down.update({Max: max_best_dec_with_meu})
    _node_functions_bottom_up = node_functions[2].copy()
    print(f'_node_functions_bottom_up {_node_functions_bottom_up}')

    nodes = get_nodes_by_type(node)

    lls_per_node = np.zeros((data.shape[0], len(nodes)))

    # one pass bottom up evaluating the likelihoods
    log_likelihood(node, data, dtype=data.dtype, node_log_likelihood=_node_functions_bottom_up, lls_matrix=lls_per_node) # node_log_likelihood=_node_functions_bottom_up

    meu_per_node = np.zeros((data.shape[0], len(nodes)))
    # one pass for meu
    meu(node, data, meu_matrix=meu_per_node)

    instance_ids = np.arange(data.shape[0])

    # one pass top down to decide on the max branch until it reaches a leaf, then it fills the nan slot with the mode
    eval_spmn_top_down(node, _node_functions_top_down, parent_result=instance_ids, data=data, lls_per_node=lls_per_node, meu_per_node=meu_per_node)

    return data
