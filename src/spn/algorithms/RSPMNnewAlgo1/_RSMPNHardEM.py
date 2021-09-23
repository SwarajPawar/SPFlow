import copy
import logging

import numpy as np

from spn.algorithms.Inference import log_likelihood
from spn.algorithms.MPE import get_node_funtions
from spn.algorithms.RSPMN.TemplateUtil import eval_template_top_down
from spn.structure.Base import get_nodes_by_type


def parallelise(self, data):
    # Parallelizing using Pool.apply()

    import multiprocessing as mp

    print("in method parralelise()")
    # Step 1: Init multiprocessing.Pool()
    pool = mp.Pool(mp.cpu_count())

    print(f'mp.cpu_count() {mp.cpu_count()}')
    # Step 2: `pool.apply` the `howmany_within_range()`
    import time
    start_time = time.time()
    unrolled_network_lls_per_node = [
        pool.apply(
            self.eval_rspmn_bottom_up,
            args=(self.template, row.reshape(1, -1), True)) for row in data
    ]

    # Step 3: Don't forget to close
    pool.close()
    print("--- %s seconds ---" % (time.time() - start_time))
    print(f'unrolled_network_lls_per_node {unrolled_network_lls_per_node}')


def hard_em(self, data, template, total_num_of_time_steps_varies=False):
    """
    Learns the final template structure by optimising the weights on
    bottom sum interface node by incrementing counts.
    :return: Final template structure
    Note: Optimises weights only on bottom sum interface node.
    Can be modified to update weights on all sum nodes in update_weights method
    """
    self.template = copy.deepcopy(template)

    node_functions = get_node_funtions()
    _node_functions_top_down = node_functions[0].copy()
    # _node_functions_top_down.update({Sum: self.mpe_interface_sum})
    logging.debug(f'_node_functions_top_down {_node_functions_top_down}')

    if total_num_of_time_steps_varies:
        assert type(data) is list, 'When sequence length varies, data is ' \
                                   'a list of numpy arrays'
        print('Evaluating rspmn')

        # self.parallelise(data)

        import time
        start_time = time.time()
        for row in range(len(data)):
            # print(f'row {row}')
            each_data_point = data[row]
            # print("length of sequence:", self.get_len_sequence())
            each_data_point = each_data_point.reshape(1, -1)
            unrolled_network_lls_per_node = self.eval_rspmn_bottom_up(
                self.template, each_data_point, True
            )

            self.eval_rspmn_top_down(
                self.template, each_data_point, unrolled_network_lls_per_node,
                _node_functions_top_down
            )
        print("--- %s seconds ---" % (time.time() - start_time))
    else:
        assert type(data) is np.ndarray, 'data should be of type numpy' \
                                         ' array'

        # one pass bottom up to get likelihoods
        unrolled_network_lls_per_node = self.eval_rspmn_bottom_up(
            self.template, data
        )
        # one pass top down to increment counts
        self.eval_rspmn_top_down(
            self.template, data, unrolled_network_lls_per_node,
            _node_functions_top_down
        )

    # update weights on bottom sum interface node
    # self.update_weights(self.template)
    return self.template


def eval_rspmn_bottom_up(self, template, data):
    """
    Evaluates network bottom up by passing the likelihoods
    from bottom time step to top
    :return:unrolled_network_eval_val_per_node:
    list of likelihoods of networks corresponding to each time step
    starting with bottom most to top network.
    Note: unrolled_network_eval_val_per_node stores
    dummy bottom + 1 time step likelihoods as 0th element in list
    """
    # assert self.InitialTemplate.top_network is not None,
    # f'top layer does not exist'
    # assert self.template is not None, f'template layer does not exist'
    # print(type(data))
    logging.debug(f'in method eval_rspmn_bottom_up()')

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

    # for bottom most time step + 1
    eval_val_per_node = np.zeros((data.shape[0], len(template_nodes)))
    unrolled_network_eval_val_per_node = [eval_val_per_node]

    # evaluate template bottom up at each time step
    for time_step_num_in_reverse_order in range(total_num_of_time_steps - 1,
                                                -1, -1):

        print(f'time_step_num_in_reverse_order '
            f'{time_step_num_in_reverse_order}')
        logging.debug(
            f'time_step_num_in_reverse_order '
            f'{time_step_num_in_reverse_order}')

        prev_eval_val_per_node = unrolled_network_eval_val_per_node[-1]
        logging.debug(f'prev_eval_val_per_node {prev_eval_val_per_node.shape}')

        # attach likelihoods of bottom interface root nodes as
        # data for latent leaf vars
        each_time_step_data_for_template = \
            self.get_each_time_step_data_for_template(
                data,
                time_step_num_in_reverse_order,
                total_num_of_time_steps,
                prev_eval_val_per_node,
                initial_num_latent_interface_nodes,
                num_variables_each_time_step,
                bottom_up=True
            )

        # if time step is 0, evaluate top network
        if time_step_num_in_reverse_order == 0:
            top_nodes = get_nodes_by_type(self.InitialTemplate.top_network)

            eval_val_per_node = np.zeros((data.shape[0], len(top_nodes)))
            log_likelihood(self.InitialTemplate.top_network,
                           each_time_step_data_for_template,
                           lls_matrix=eval_val_per_node)
        else:
            eval_val_per_node = np.zeros((data.shape[0], len(template_nodes)))
            log_likelihood(template, each_time_step_data_for_template,
                           lls_matrix=eval_val_per_node)

        unrolled_network_eval_val_per_node.append(eval_val_per_node)

    # print(np.mean(unrolled_network_eval_val_per_node[-1][:, 0]))

    return unrolled_network_eval_val_per_node


def eval_rspmn_top_down(self, template, data,
                        unrolled_network_lls_per_node,
                        node_functions_top_down=None
                        ):
    """
    Increments the counts of nodes by passing data through a
    specific branch based on likelihood
    :param unrolled_network_lls_per_node: likelihoods for each time
    step obtained by bottom up pass
    """
    logging.debug(f'in method eval_rspmn_top_down()')

    num_variables_each_time_step, total_num_of_time_steps, \
        initial_num_latent_interface_nodes = \
        self.get_params_for_get_each_time_step_data_for_template(template,
                                                                 data)

    # Should not increment count on Latent interface leafs on last time step
    # so total_num_of_time_steps - 1
    for time_step_num in range(total_num_of_time_steps - 1):
        lls_per_node = unrolled_network_lls_per_node[
            total_num_of_time_steps - time_step_num
            ]

        # attach likelihoods of bottom interface root nodes as
        # data for latent leaf vars
        each_time_step_data_for_template = \
            self.get_each_time_step_data_for_template(
                data, time_step_num,
                total_num_of_time_steps,
                lls_per_node,
                initial_num_latent_interface_nodes,
                num_variables_each_time_step,
                bottom_up=False
            )

        instance_ids = np.arange(each_time_step_data_for_template.shape[0])
        # if time step is 0, evaluate top network
        if time_step_num == 0:
            all_results, latent_interface_dict = eval_template_top_down(
                self.InitialTemplate.top_network,
                node_functions_top_down, False,
                all_results=None, parent_result=instance_ids,
                data=each_time_step_data_for_template,
                lls_per_node=lls_per_node)

        else:
            all_results, latent_interface_dict = eval_template_top_down(
                template,
                node_functions_top_down, False,
                all_results=None, parent_result=instance_ids,
                data=each_time_step_data_for_template,
                lls_per_node=lls_per_node
            )

        # initialise template.interface_winner.
        # Each instance must reach one leaf interface node.
        # Initial interface node number is infinite
        template.interface_winner = np.full(
            (each_time_step_data_for_template.shape[0],), np.inf
        )
        logging.debug(f'latent_interface_dict {latent_interface_dict}')

        # for each instance assign the interface node reached
        for latent_interface_node, instances in \
                latent_interface_dict.items():
            template.interface_winner[instances] = \
                latent_interface_node.interface_idx - \
                num_variables_each_time_step

        # if self.template.interface_winner.any(np.inf):
        #     raise Exception(f'All instances are not passed to
        #     the corresponding latent interface nodes')
