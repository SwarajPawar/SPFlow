from scipy.special import logsumexp

from spn.structure.Base import get_topological_order_layers, Leaf
import numpy as np

from spn.structure.leaves.spmnLeaves.SPMNLeaf import LatentInterface

from spn.structure.Base import Max


def eval_template_top_down(root, eval_functions, soft_em=False,
                           all_results=None, parent_result=None,
                           data=None, lls_per_node=None, meu_per_node=None,
                           **args):
    """
    evaluates an spn top to down


    :param root: spnt root
    :param eval_functions: is a dictionary that contains k:Class of the node, v:lambda function that receives as parameters (node, [parent_results], args**) and returns {child : intermediate_result}. This intermediate_result will be passed to child as parent_result. If intermediate_result is None, no further propagation occurs
    :param all_results: is a dictionary that contains k:Class of the node, v:result of the evaluation of the lambda function for that node.
    :param parent_result: initial input to the root node
    :param args: free parameters that will be fed to the lambda functions.
    :return: the result of computing and propagating all the values throught the network
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
    latent_interface_dict = {}

    for layer in reversed(get_topological_order_layers(root)):
        for n in layer:
            func = n.__class__._eval_func[-1]

            param = all_results[n]

            if type(n) == Max and meu_per_node is not None:
                result = func(n, param, data=data, meu_per_node=meu_per_node)
            else:
                result = func(n, param, data=data, lls_per_node=lls_per_node)

            if soft_em:  # soft em
                if type(n) == LatentInterface:
                    top_down_pass_val=logsumexp(np.concatenate(param).reshape(-1, 1),
                              axis=1)
                    latent_interface_dict[n] = top_down_pass_val
            else:  # hard em
                top_down_pass_val = np.concatenate(param)
                if len(top_down_pass_val) > 0:
                    n.count = n.count + len(top_down_pass_val)
                    if type(n) == LatentInterface:
                        latent_interface_dict[n] = top_down_pass_val

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

    return all_results[root], latent_interface_dict


def gradient_backward(spn, lls_per_node, node_gradients=None, data=None):
    gradient_result = np.zeros_like(lls_per_node)
    soft_em = True

    all_results, latent_interface_dict = eval_template_top_down(
        spn,
        node_gradients, soft_em,
        parent_result=np.zeros((lls_per_node.shape[0])),
        gradient_result=gradient_result,
        lls_per_node=lls_per_node,
        data=data
    )

    return gradient_result, latent_interface_dict






# def mpe_interface_sum(self, node, parent_result, data=None, lls_per_node=None,
    #             rand_gen=None):
    #     if parent_result is None:
    #         return None
    #
    #     parent_result = np.concatenate(parent_result)
    #
    #     # logging.debug(f'sum node {node}')
    #     if all(isinstance(child, LatentInterface) for child in
    #            node.children):
    #
    #         w_children_log_probs = np.zeros(
    #             (len(parent_result), len(node.weights)))
    #         for i, c in enumerate(node.children):
    #             w_children_log_probs[:, i] = \
    #                 lls_per_node[parent_result, c.id] + np.log(node.weights[i])
    #
    #         print(f'w_children_log_probs {w_children_log_probs}')
    #         # max_child_branches = np.argmax(w_children_log_probs, axis=1)
    #
    #         print(f'data {data}')
    #         interface_idx = self.mpe_for_latent_interface_from_top_network(data[parent_result])
    #         max_child_branches = interface_idx[~np.isnan(interface_idx)]
    #         max_child_branches = max_child_branches - len(self.params.feature_names)
    #         print(f'max_child_branches {max_child_branches}')
    #         children_row_ids = {}
    #
    #         for i, c in enumerate(node.children):
    #             children_row_ids[c] = parent_result[max_child_branches == i]
    #
    #     else:
    #         w_children_log_probs = np.zeros(
    #             (len(parent_result), len(node.weights)))
    #         for i, c in enumerate(node.children):
    #             w_children_log_probs[:, i] = lls_per_node[
    #                                              parent_result, c.id] + np.log(
    #                 node.weights[i])
    #
    #         max_child_branches = np.argmax(w_children_log_probs, axis=1)
    #
    #         children_row_ids = {}
    #
    #         for i, c in enumerate(node.children):
    #             children_row_ids[c] = parent_result[max_child_branches == i]
    #
    #     return children_row_ids
    #
    # def mpe_for_latent_interface_from_top_network(self, data):
    #
    #     mpe_data = data[:, 0:len(self.params.feature_names)].copy()
    #     rows = data.shape[0]
    #     columns = data.shape[1] - len(self.params.feature_names)
    #     nan_data = np.full((rows, columns), np.nan)
    #     print(f'nan_data shape {nan_data.shape}')
    #     print(f'mpe_data shape {mpe_data.shape}')
    #     mpe_data = np.column_stack((mpe_data, nan_data))
    #     mpe_data = mpe(self.InitialTemplate.top_network, mpe_data)
    #     print(f'mpe data {mpe_data[0:10]}')
    #     interface_idx = mpe_data[:, len(self.params.feature_names):]
    #     print(f'interface_idx[0:100] {interface_idx[0:10]}')
    #     return interface_idx
