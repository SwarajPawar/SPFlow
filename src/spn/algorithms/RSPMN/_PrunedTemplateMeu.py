import logging

import numpy as np

import spn.algorithms.MEU as spmnMeu
from spn.algorithms.MEUTopDown import max_best_dec_with_meu
from spn.algorithms.MPE import get_node_funtions, mpe_sum_using_likelihoods
from spn.algorithms.RSPMN.TemplateUtil import eval_template_top_down
from spn.structure.Base import Max, Sum
from spn.structure.Base import get_nodes_by_type
from spn.structure.leaves.spmnLeaves.SPMNLeaf import LatentInterface



def value_iteration(self, template, iterations):

    num_variables_each_time_step = len(self.params.feature_names)

    data = [np.nan]*(num_variables_each_time_step*iterations)
    data = np.array(data).reshape(1, -1)

    unrolled_network_meu_per_node, unrolled_network_likelihood_per_node = \
        self.eval_rspmn_bottom_up_for_meu(template, data)

    return unrolled_network_meu_per_node, unrolled_network_likelihood_per_node


def eval_rspmn_bottom_up_for_meu(self, template, data):
    """
    :return: unrolled_network_meu_per_node, unrolled_network_likelihood_per_node
    list of meus/likelihoods of networks corresponding to each time step
    starting with bottom most to top network.
    Note: stores dummy bottom + 1 time step meu/likelihoods
    as 0th element in list
    """
    # assert self.InitialTemplate.top_network is not None,
    # f'top layer does not exist'
    # assert self.template is not None, f'template layer does not exist'

    assert type(data) is np.ndarray, 'data should be of type numpy array'

    num_variables_each_time_step, total_num_of_time_steps, \
        initial_num_latent_interface_nodes = \
        self.get_params_for_get_each_time_step_data_for_template(template,
                                                                 data)

    logging.debug(
        f'intial_num_latent_interface_nodes '
        f'{initial_num_latent_interface_nodes}')
    logging.debug(f'total_num_of_time_steps {total_num_of_time_steps}')

    template_nodes = get_nodes_by_type(template)
    latent_interface_list = []
    for node in template_nodes:
        if type(node) == LatentInterface:
            latent_interface_list.append(node)


    # for bottom most time step + 1
    likelihood_per_node = np.zeros((data.shape[0], len(template_nodes)))
    unrolled_network_likelihood_per_node = [likelihood_per_node]

    meu_per_node = np.zeros((data.shape[0], len(template_nodes)))
    unrolled_network_meu_per_node = [meu_per_node]

    # evaluate template bottom up at each time step
    for time_step_num_in_reverse_order in range(total_num_of_time_steps - 1,
                                                -1, -1):

        logging.debug(
            f'time_step_num_in_reverse_order '
            f'{time_step_num_in_reverse_order}')

        prev_likelihood_per_node = unrolled_network_likelihood_per_node[-1]
        logging.debug(f'prev_likelihood_per_node {prev_likelihood_per_node.shape}')
        #print(f'prev_likelihood_per_node {prev_likelihood_per_node}')

        prev_meu_per_node = unrolled_network_meu_per_node[-1]
        logging.debug(f'prev_meu_per_node {prev_meu_per_node.shape}')
        #print(f'prev_meu_per_node {prev_meu_per_node}')

        # attach likelihoods of bottom interface root nodes as
        # data for latent leaf vars
        each_time_step_data_for_template = \
            self.get_each_time_step_data_for_meu_pruned_template(
                data,
                time_step_num_in_reverse_order,
                total_num_of_time_steps,
                prev_likelihood_per_node,
                initial_num_latent_interface_nodes,
                num_variables_each_time_step,
                bottom_up=True
            )
        # if time step is 0, evaluate top network
        if time_step_num_in_reverse_order == 0:

            # print(
            #     f'each_time_step_data_for_template: {each_time_step_data_for_template}')

            top_nodes = get_nodes_by_type(self.InitialTemplate.top_network)
            meu_per_node = np.zeros((data.shape[0], len(top_nodes)))
            meu_per_node.fill(np.nan)
            likelihood_per_node = np.zeros((data.shape[0], len(top_nodes)))

            top_latent_interface_list = []
            for node in top_nodes:
                if type(node) == LatentInterface:
                    top_latent_interface_list.append(node)

            # replace values of latent leaf nodes with
            # bottom time step meu values
            self.pass_meu_val_to_latent_interface_leaf_nodes_pruned_template(
                meu_per_node, prev_meu_per_node,
                initial_num_latent_interface_nodes, top_latent_interface_list)

            #print(f'initial meu_per_node {meu_per_node}')

            spmnMeu.meu(self.InitialTemplate.top_network,
                        each_time_step_data_for_template,
                        meu_matrix=meu_per_node,
                        lls_matrix=likelihood_per_node
                                     )

            # eval_val_per_node = meu_matrix
            #print(f'meu_per_node {meu_per_node}')
            # print(f'likelihood_per_node {likelihood_per_node}')
            # print(f'meu_matrix {meu_matrix}')

        else:
            meu_per_node = np.zeros((data.shape[0], len(template_nodes)))
            meu_per_node.fill(np.nan)
            likelihood_per_node = np.zeros((data.shape[0], len(template_nodes)))

            # replace values of latent leaf nodes with
            # bottom time step meu values
            self.pass_meu_val_to_latent_interface_leaf_nodes_pruned_template(
                meu_per_node, prev_meu_per_node,
                initial_num_latent_interface_nodes, latent_interface_list)
            #print(f'initial meu_per_node {meu_per_node}')
            spmnMeu.meu(template,
                        each_time_step_data_for_template,
                        meu_matrix=meu_per_node,
                        lls_matrix=likelihood_per_node
                                     )

            # meu_per_node = meu_matrix
            #print(f'meu_per_node {meu_per_node}')
            # print(f'likelihood_per_node {likelihood_per_node}')

        unrolled_network_likelihood_per_node.append(likelihood_per_node)
        unrolled_network_meu_per_node.append(meu_per_node)

    # print(unrolled_network_meu_per_node[-1][:, 0])

    return unrolled_network_meu_per_node, unrolled_network_likelihood_per_node


def select_actions(self, template, data,
                   unrolled_network_meu_per_node,
                   unrolled_network_likelihood_per_node):
    node_functions = get_node_funtions()
    node_functions_top_down = node_functions[0].copy()
    node_functions_top_down.update({Max: max_best_dec_with_meu})
    node_functions_top_down.update({Sum: mpe_sum_using_likelihoods})

    # node_functions_bottom_up = node_functions[2].copy()
    # logging.debug(f'_node_functions_bottom_up {node_functions_bottom_up}')

    num_variables_each_time_step, total_num_of_time_steps, \
    initial_num_latent_interface_nodes = \
        self.get_params_for_get_each_time_step_data_for_template(template,
                                                                 data)
    # top down traversal
    time_step_num = 0
    meu_per_node, likelihood_per_node = self.meu_of_state(template, data,
                 unrolled_network_meu_per_node,
                 unrolled_network_likelihood_per_node)

    #print(f'lls_per_node {likelihood_per_node}')
    #print(f'meu_per_node {meu_per_node}')
    prev_likelihood_per_node = unrolled_network_likelihood_per_node[-2]
    each_time_step_data_for_template = \
        self.get_each_time_step_data_for_meu_pruned_template(
            data, time_step_num,
            total_num_of_time_steps,
            prev_likelihood_per_node,
            initial_num_latent_interface_nodes,
            num_variables_each_time_step,
            bottom_up=False
        )

    instance_ids = np.arange(each_time_step_data_for_template.shape[0])

    # if time step is 0, evaluate top network
    eval_template_top_down(
        self.InitialTemplate.top_network,
        node_functions_top_down, False,
        all_results=None, parent_result=instance_ids,
        meu_per_node=meu_per_node,
        data=each_time_step_data_for_template,
        lls_per_node=likelihood_per_node)

    # fill data with values returned by filling each time step data through
    # top down traversal
    data[
    :,
    (time_step_num * num_variables_each_time_step):
    (time_step_num * num_variables_each_time_step) +
    num_variables_each_time_step
    ] = \
        each_time_step_data_for_template[:, 0:num_variables_each_time_step]

    # print(data)
    return data


def meu_of_state(self, template, data,
                 unrolled_network_meu_per_node,
                 unrolled_network_likelihood_per_node):
    assert type(data) is np.ndarray, 'data should be of type numpy array'

    num_variables_each_time_step, total_num_of_time_steps, \
    initial_num_latent_interface_nodes = \
        self.get_params_for_get_each_time_step_data_for_template(template,
                                                                 data)

    logging.debug(
        f'intial_num_latent_interface_nodes '
        f'{initial_num_latent_interface_nodes}')
    logging.debug(f'total_num_of_time_steps {total_num_of_time_steps}')

    template_nodes = get_nodes_by_type(template)
    latent_interface_list = []
    for node in template_nodes:
        if type(node) == LatentInterface:
            latent_interface_list.append(node)

    time_step_num_in_reverse_order = 0

    logging.debug(
        f'time_step_num_in_reverse_order '
        f'{time_step_num_in_reverse_order}')

    prev_likelihood_per_node = unrolled_network_likelihood_per_node[-2]
    logging.debug(
        f'prev_likelihood_per_node {prev_likelihood_per_node.shape}')
    #print(f'prev_likelihood_per_node {prev_likelihood_per_node}')

    prev_meu_per_node = unrolled_network_meu_per_node[-2]
    logging.debug(f'prev_meu_per_node {prev_meu_per_node.shape}')
    #print(f'prev_meu_per_node {prev_meu_per_node}')

    # attach likelihoods of bottom interface root nodes as
    # data for latent leaf vars
    each_time_step_data_for_template = \
        self.get_each_time_step_data_for_meu(
            data,
            time_step_num_in_reverse_order,
            total_num_of_time_steps,
            prev_likelihood_per_node,
            initial_num_latent_interface_nodes,
            num_variables_each_time_step,
            bottom_up=True
        )
    # if time step is 0, evaluate top network


    # print(
    #     f'each_time_step_data_for_template: {each_time_step_data_for_template}')

    top_nodes = get_nodes_by_type(self.InitialTemplate.top_network)
    meu_per_node = np.zeros((data.shape[0], len(top_nodes)))
    meu_per_node.fill(np.nan)
    likelihood_per_node = np.zeros((data.shape[0], len(top_nodes)))

    top_latent_interface_list = []
    for node in top_nodes:
        if type(node) == LatentInterface:
            top_latent_interface_list.append(node)

    # replace values of latent leaf nodes with
    # bottom time step meu values
    self.pass_meu_val_to_latent_interface_leaf_nodes(
        meu_per_node, prev_meu_per_node,
        initial_num_latent_interface_nodes, top_latent_interface_list)

    #print(f'initial meu_per_node {meu_per_node}')

    spmnMeu.meu(self.InitialTemplate.top_network,
                each_time_step_data_for_template,
                meu_matrix=meu_per_node,
                lls_matrix=likelihood_per_node
                )

    return meu_per_node, likelihood_per_node
