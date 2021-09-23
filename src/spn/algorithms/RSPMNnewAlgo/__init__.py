import numpy as np

from spn.algorithms.RSPMNnewAlgo.RSPMNInitialTemplateBuild import RSPMNInitialTemplate
from spn.algorithms.RSPMN.TemplateUtil import eval_template_top_down, \
    gradient_backward
from spn.structure.Base import Sum, assign_ids, rebuild_scopes_bottom_up
from spn.structure.Base import get_nodes_by_type
from spn.structure.leaves.spmnLeaves.SPMNLeaf import LatentInterface
import copy

from spn.structure.Base import Leaf


class RSPMNnewAlgo:

    def __init__(self, partial_order, decision_nodes, utility_nodes,
                 feature_names, meta_types,
                 cluster_by_curr_information_set=False,
                 util_to_bin=False):

        self.params = RSPMNParams(partial_order, decision_nodes, utility_nodes,
                                  feature_names, meta_types,
                                  cluster_by_curr_information_set,
                                  util_to_bin
                                  )
        self.InitialTemplate = RSPMNInitialTemplate(self.params)
        self.template = None

    # Import methods of RSPMN class
    from ._RSMPNHardEM import eval_rspmn_bottom_up, \
        eval_rspmn_top_down, hard_em, parallelise
    from ._RSPMNSoftEM import rspmn_gradient_backward, soft_em
    from ._RSPMNMeu import meu, eval_rspmn_bottom_up_for_meu,\
        topdowntraversal_and_bestdecisions, value_iteration, \
        select_actions, meu_of_state
    from ._PrunedTemplateMeu import value_iteration, \
        eval_rspmn_bottom_up_for_meu, select_actions, meu_of_state

    # Other methods of class
    def get_params_for_get_each_time_step_data_for_template(self,
                                                            template, data):

        num_variables_each_time_step = len(self.params.feature_names)
        total_num_of_time_steps = int(
            data.shape[1] / num_variables_each_time_step)
        initial_num_latent_interface_nodes = len(template.children)

        return num_variables_each_time_step, \
               total_num_of_time_steps, initial_num_latent_interface_nodes

    @staticmethod
    def get_each_time_step_data_for_template(data, time_step_num,
                                             total_num_of_time_steps,
                                             eval_val_per_node,
                                             initial_num_latent_interface_nodes,
                                             num_variables_each_time_step,
                                             bottom_up=True):

        if time_step_num == -1:

            each_time_step_data = \
                np.empty((data.shape[0], num_variables_each_time_step))
            each_time_step_data[:] = np.nan

        else:
            each_time_step_data = data[:,
                                  (time_step_num * num_variables_each_time_step):
                                  (time_step_num * num_variables_each_time_step) +
                                  num_variables_each_time_step]

        assert each_time_step_data.shape[1] == num_variables_each_time_step

        if time_step_num == total_num_of_time_steps - 1:
            # last time step in the sequence. last level of template
            # latent node data is corresponding bottom time step interface
            # node's inference value. 1 for last level
            latent_node_data = np.zeros((each_time_step_data.shape[0],
                                         initial_num_latent_interface_nodes))

        else:
            if bottom_up:
                # latent node data is corresponding bottom time step
                # interface node's inference value
                latent_node_data = eval_val_per_node[:,
                                   1:initial_num_latent_interface_nodes + 1]
            else:
                # latent node data top down is curr time step lls_
                # per_node last set of values
                first_latent_node_column = \
                    eval_val_per_node.shape[1] - \
                    initial_num_latent_interface_nodes
                latent_node_data = \
                    eval_val_per_node[:, first_latent_node_column:]

        each_time_step_data_for_template = np.concatenate(
            (each_time_step_data, latent_node_data), axis=1)

        return each_time_step_data_for_template



    @staticmethod
    def get_each_time_step_data_for_meu_pruned_template(
         data, time_step_num, total_num_of_time_steps, eval_val_per_node,
         initial_num_latent_interface_nodes,
         num_variables_each_time_step,
         bottom_up=True
    ):

        if time_step_num == -1:

            each_time_step_data = \
                np.empty((data.shape[0], num_variables_each_time_step))
            each_time_step_data[:] = np.nan

        else:
            each_time_step_data = \
                data[:, (time_step_num * num_variables_each_time_step):
                        (time_step_num * num_variables_each_time_step) +
                        num_variables_each_time_step
                ]

        assert each_time_step_data.shape[1] == num_variables_each_time_step

        if time_step_num == total_num_of_time_steps - 1:
            # last time step in the sequence. last level of template
            # latent node data is corresponding bottom time step interface
            # node's inference value. 1 for last level
            latent_node_data = np.zeros((each_time_step_data.shape[0],
                                         initial_num_latent_interface_nodes))

        else:
            if bottom_up:
                # latent node data is corresponding bottom time step
                # interface node's inference value
                latent_node_data = eval_val_per_node[:,
                                   1:initial_num_latent_interface_nodes + 1]
                latent_node_data = np.log(latent_node_data)
            else:
                # latent node data is corresponding bottom time step
                # interface node's inference value
                latent_node_data = eval_val_per_node[:,
                                   1:initial_num_latent_interface_nodes + 1]
                latent_node_data = np.log(latent_node_data)

        each_time_step_data_for_template = np.concatenate(
            (each_time_step_data, latent_node_data), axis=1)

        return each_time_step_data_for_template

    @staticmethod
    def get_each_time_step_data_for_meu(data, time_step_num,
                                             total_num_of_time_steps,
                                             eval_val_per_node,
                                             initial_num_latent_interface_nodes,
                                             num_variables_each_time_step,
                                             bottom_up=True):

        each_time_step_data = data[:,
                              (time_step_num * num_variables_each_time_step):
                              (
                                      time_step_num * num_variables_each_time_step) + num_variables_each_time_step]

        assert each_time_step_data.shape[1] == num_variables_each_time_step

        if time_step_num == total_num_of_time_steps - 1:
            # last time step in the sequence. last level of template
            # latent node data is corresponding bottom time step interface
            # node's inference value. 1 for last level
            latent_node_data = np.zeros((each_time_step_data.shape[0],
                                         initial_num_latent_interface_nodes))

        else:
            if bottom_up:
                # latent node data is corresponding bottom time step
                # interface node's inference value
                latent_node_data = eval_val_per_node[:,
                                   1:initial_num_latent_interface_nodes + 1]
                latent_node_data = np.log(latent_node_data)
            else:
                # latent node data top down is curr time step lls_
                # per_node last set of values
                first_latent_node_column = \
                    eval_val_per_node.shape[1] - \
                    initial_num_latent_interface_nodes
                latent_node_data = \
                    eval_val_per_node[:, first_latent_node_column:]

        each_time_step_data_for_template = np.concatenate(
            (each_time_step_data, latent_node_data), axis=1)

        return each_time_step_data_for_template

    def pass_meu_val_to_latent_interface_leaf_nodes(self,
            eval_val_per_node, prev_eval_val_per_node,
            num_of_template_children,
            latent_node_list
    ):
        num_variables_each_time_step = len(self.params.feature_names)
        for i in range(0, len(latent_node_list),num_of_template_children):
            for j in range(i, i+num_of_template_children):
                k = j % num_of_template_children
                #print(f"latent_node_list[j].interface_idx {latent_node_list[j].interface_idx}")
                l = latent_node_list[j].interface_idx-num_variables_each_time_step
                #print(f"k,l: {k, l}")
                # print(f'latent_node_list[j].id in pass meu {latent_node_list[j].id}')
                eval_val_per_node[:, latent_node_list[j].id] = \
                    prev_eval_val_per_node[:, k+1]

        # for latent_node in latent_node_list:
        #     k = latent_node.interface_idx-num_variables_each_time_step
        #     eval_val_per_node[:, latent_node.id] = \
        #         prev_eval_val_per_node[:, k + 1]
        # # leaf latent node columns in eval_val_per_node correspond to
        # # last few columns in current time step. They are equal to
        # # num of latent interface nodes
        # first_latent_node_column = \
        #     eval_val_per_node.shape[1] - \
        #     num_of_template_children
        #
        # # eval val of last columns corr to latent interface nodes of
        # # current time step are equal to first columns corr to
        # # eval vals of prev time step
        # eval_val_per_node[:, first_latent_node_column:] = \
        #     prev_eval_val_per_node[:, 1:num_of_template_children + 1]
        # print(f'meu at pass meu {eval_val_per_node}')
        return eval_val_per_node

    def pass_meu_val_to_latent_interface_leaf_nodes_pruned_template(self,
            eval_val_per_node, prev_eval_val_per_node,
            num_of_template_children,
            latent_node_list
    ):
        num_variables_each_time_step = len(self.params.feature_names)

        for latent_node in latent_node_list:
            k = latent_node.interface_idx-num_variables_each_time_step
            # print(f'(k,id): {k, latent_node.id}')
            # print(f"eval_val_per_node.shape {eval_val_per_node.shape}")
            # print(f"prev_eval_val_per_node.shape {prev_eval_val_per_node.shape}")
            eval_val_per_node[:, latent_node.id] = \
                prev_eval_val_per_node[:, k + 1]

        return eval_val_per_node

    @staticmethod
    def update_weights(template):
        """
        Updates weights on bottom sum interface node
        """

        nodes = get_nodes_by_type(template)

        for node in nodes:

            if isinstance(node, Sum):

                # if all(isinstance(child, LatentInterface) for child in
                #        node.children):

                    for i, child in enumerate(node.children):
                        node.weights[i] = (node.children[i].count / node.count)

                    node.weights = (np.array(node.weights) / np.sum(
                        node.weights)).tolist()

                    #print(node.weights)

            #print(f'node {node}, count {node.count}')

    @staticmethod
    def prune_latent_interface_nodes(template):
        """
        Updates weights on bottom sum interface node
        """

        nodes = get_nodes_by_type(template)
        #
        # for node in nodes:
        #
        #     if isinstance(node, Sum):
        #
        #         if all(isinstance(child, LatentInterface) for child in
        #                node.children):
        #             child_node_weights = []
        #             remove_children = []
        #             # node_children = copy.deepcopy(node.children)
        #             for i, child in enumerate(node.children):
        #                 if node.children[i].count == 1:
        #                     remove_children.append(child)
        #                 else:
        #                     child_node_weights.append(node.weights[i])
        #
        #             remaining_nodes = [node for node in node.children
        #                                if node not in remove_children]
        #
        #             node.children = remaining_nodes
        #
        #             node.weights = child_node_weights
        #
        #             if node.weights:
        #                 node.weights = (np.array(node.weights) / np.sum(
        #                     node.weights)).tolist()

                    #print(node.weights)

            #print(f'node {node}, count {node.count}')


        for node in nodes:

            remove_children = []

            if not isinstance(node, Leaf):
                for child in node.children:

                    if isinstance(child, Sum):

                        # if all(isinstance(grand_child, LatentInterface) for grand_child in
                        #        child.children):
                            child_node_weights = []
                            remove_grand_children = []
                            # node_children = copy.deepcopy(node.children)
                            for i, grand_child in enumerate(child.children):
                                if child.children[i].count == 1:
                                    remove_grand_children.append(grand_child)
                                else:
                                    child_node_weights.append(child.weights[i])

                            remaining_nodes = [node for node in child.children
                                               if node not in remove_grand_children]

                            child.children = remaining_nodes

                            child.weights = child_node_weights

                            if child.weights:
                                child.weights = (np.array(child.weights) / np.sum(
                                    child.weights)).tolist()

                            if not child.children:
                                remove_children.append(child)

                remaining_child_nodes = [node for node in node.children
                                   if node not in remove_children]

                node.children = remaining_child_nodes

        assign_ids(template)
        rebuild_scopes_bottom_up(template)

        #nodes = get_nodes_by_type(template)





    def log_likelihood(self, template, data):

        unrolled_network_lls_per_node = self.eval_rspmn_bottom_up(template,
                                                                  data, True)
        # ll at root node
        log_likelihood = unrolled_network_lls_per_node[-1][:, 0]

        return log_likelihood, unrolled_network_lls_per_node


class RSPMNParams:

    def __init__(self, partial_order, decision_nodes, utility_nodes,
                 feature_names, meta_types,
                 cluster_by_curr_information_set, util_to_bin):
        self.partial_order = partial_order
        self.decision_nodes = decision_nodes
        self.utility_nodes = utility_nodes
        self.feature_names = feature_names
        self.meta_types = meta_types
        self.cluster_by_curr_information_set = cluster_by_curr_information_set
        self.length_of_time_slice = len(self.feature_names)
        self.util_to_bin = util_to_bin
