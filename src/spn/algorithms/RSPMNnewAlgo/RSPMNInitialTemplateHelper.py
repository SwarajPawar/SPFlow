# two time step utilities
import logging


def get_partial_order_two_time_steps(partial_order_one_time_step):

    partial_order_time_step_0 = []
    partial_order_time_step_1 = []
    for information_set in partial_order_one_time_step:
        logging.debug(f'information_set {information_set}')

        information_set_time_step_0 = []
        information_set_time_step_1 = []
        for var in information_set:
            information_set_time_step_0.append(var + 'T0')
            information_set_time_step_1.append(var + 'T1')

        partial_order_time_step_0.append(information_set_time_step_0)
        partial_order_time_step_1.append(information_set_time_step_1)

    partial_order_two_time_steps = partial_order_time_step_0 + partial_order_time_step_1

    logging.debug(f'partial_order_two_time_steps {partial_order_two_time_steps}')

    return partial_order_two_time_steps


def get_feature_names_two_time_steps(partial_order_two_time_steps):

    feature_names_two_time_steps = [var for var_set in partial_order_two_time_steps for var in var_set]

    return feature_names_two_time_steps


def get_nodes_two_time_steps(nodes_list):
    nodes_time_step_0 = []
    nodes_time_step_1 = []  
    for var in nodes_list:
        nodes_time_step_0.append(var + 'T0')
        nodes_time_step_1.append(var + 'T1')

    nodes_two_time_steps = nodes_time_step_0 + nodes_time_step_1
 
    logging.debug(f'nodes_two_time_steps {nodes_two_time_steps}')

    return nodes_two_time_steps


def get_meta_types_two_time_steps(meta_types):

    meta_types_two_time_steps = meta_types + meta_types
    return meta_types_two_time_steps
