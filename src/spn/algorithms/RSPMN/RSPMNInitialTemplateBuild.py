import numpy as np
import logging
import collections
import copy
from spn.structure.Base import Leaf, Product, Sum, InterfaceSwitch, assign_ids

from spn.algorithms.SPMN import SPMN, SPMNParams

from spn.algorithms.RSPMN.RSPMNInitialTemplateHelper import get_partial_order_two_time_steps, \
    get_feature_names_two_time_steps \
    , get_nodes_two_time_steps, get_meta_types_two_time_steps

from spn.structure.leaves.spmnLeaves.SPMNLeaf import LatentInterface

from spn.structure.Base import rebuild_scopes_bottom_up

from spn.structure.Base import Max


class RSPMNInitialTemplate:

    def __init__(self, rspmn_params):
        self.two_time_step_params = RSPMNTwoTimeStepParams(rspmn_params)
        self.spmn_structure_two_time_steps = None
        self.top_network = None
        self.template_network = None

    def build_initial_template(self, data):
        """
        :param data: data for two time steps
        :return: spmn for two time steps, top network, interface network
        """

        spmn_structure_two_time_steps = self.learn_spmn_two_time_steps(data)

        self.spmn_structure_two_time_steps = copy.deepcopy(
            spmn_structure_two_time_steps)

        logging.debug(
            f'length_of_time_slice '
            f'{self.two_time_step_params.length_of_each_time_slice}')

        # scope of each random variable in the SPMN is its position
        # in feature_names which follows partial order

        # top interface parent's scope is utility0
        # scope plus scope of time slice 1

        scope_top_interface_parent = [
            scope for scope, var in
            enumerate(
                self.two_time_step_params.feature_names_two_time_steps
            )
            if var not in
            self.two_time_step_params.decision_nodes_two_time_steps and
            scope >= self.two_time_step_params.length_of_each_time_slice-1
        ]

        logging.debug(
            f'scope_top_interface_parent {scope_top_interface_parent}')

        self.top_network, self.template_network = \
            self.get_top_and_interface_networks(
                spmn_structure_two_time_steps,
                scope_top_interface_parent
            )
        # self.top_network = top_network
        # self.template_network = template_network

        return self.spmn_structure_two_time_steps, self.top_network, \
            self.template_network

    def get_top_and_interface_networks(self, spmn_structure_two_time_steps,
                                       scope_top_interface_parent):
        """
        :param spmn_structure_two_time_steps:
        :param scope_top_interface_parent: scope of node that connects
        last variable in time slice zero and rest of the variables
        in time slice one.
        :return: top and template network sliced from input spmn for
        two time steps
        """

        interface_nodes = []
        scope_time_slice_1 = scope_top_interface_parent[1:]
        scope_time_slice_0 = list(
            range(
                0, self.two_time_step_params.length_of_each_time_slice
            )
        )
        logging.debug(f'scope_time_slice_0 {scope_time_slice_0}')

        # perform bfs
        seen, queue = set([spmn_structure_two_time_steps]), collections.deque(
            [spmn_structure_two_time_steps])
        interface_num = 0
        while queue:
            node = queue.popleft()
            if not isinstance(node, Leaf):

                time_step_1_children = []

                if isinstance(node, Product):
                    # and node.scope == scope_top_interface_parent:
                    logging.debug(f'node.scope {node} {node.scope}')
#                    logging.debug(f'child.scope {child.scope}')

                    node_children = node.children.copy()
                    for child in node_children:
                        logging.debug(f'child.scope {child} {child.scope}')

                        # check for top interface parent node
                        if not set(child.scope).intersection(
                                scope_time_slice_0):
                            time_step_1_children.append(child)
                            logging.debug(
                                f'time_step_1_children {time_step_1_children}')
                            node.children.remove(child)

                if time_step_1_children:
                    # and isinstance(node, Product) and
                    # node.scope == scope_top_interface_parent:
                    # parent interface node detected

                    # interface root node is the time step 1 root node
                    interface_node = self.make_interface_root_node(
                        time_step_1_children)
                    interface_nodes.append(interface_node)

                    # create latent interface node for
                    # corresponding interface node
                    num_features_in_one_time_step = int(len(
                        self.two_time_step_params.
                        feature_names_two_time_steps) / 2)
                    interface_idx = interface_num + \
                        num_features_in_one_time_step
                    interface_scope = node.scope[0] + 1
                    latent_interface_child = LatentInterface(
                        interface_idx=interface_idx,
                        scope=interface_scope)

                    logging.debug(
                        f'latent interface child scope '
                        f'{latent_interface_child.scope}')

                    # attach latent interface node to
                    # time step 0 interface parent
                    node.children.append(latent_interface_child)
                    interface_num += 1

                else:
                    for child in node.children:
                        # continue bfs
                        if child not in seen:
                            seen.add(child)
                            queue.append(child)

        logging.debug(f'interface_nodes {interface_nodes}')
        template_network = self.make_template_network(interface_nodes)

        top_network = spmn_structure_two_time_steps
        assign_ids(top_network)
        rebuild_scopes_bottom_up(top_network)
        return top_network, template_network

    @staticmethod
    def make_interface_root_node(children_list):
        """
        :param children_list: time step one children of time step zero
        parent node that is one level above last leaf
        variable in time step zero and time step one root.
        :return: interface node that corresponds to time step one root
        """

        logging.debug(f'time_step_1_children {children_list}')
        if len(children_list) > 1:
            # more than one child of interface parent prod
            # node belong to time step 1
            child_prod_node = Product(children=children_list)
            interface_node = child_prod_node  # copy.deepcopy(prod_node)

        else:
            interface_node = children_list[
                0]  # copy.deepcopy(time_step_1_children)

        assign_ids(interface_node)
        rebuild_scopes_bottom_up(interface_node)
        return interface_node

    def make_template_network(self, interface_nodes):
        """
        :param interface_nodes: list of interface nodes
        :return: template network with interface switch as root of
         interface nodes and interface nodes connected
        at bottom for next time step
        """

        # interface switch scope is equal to its child interface
        # node's scope, all of its children have same scope

        template_network = InterfaceSwitch(children=interface_nodes)
        template_network = self.\
            attach_interface_nodes_to_template_network_at_bottom(
                template_network
            )
        assign_ids(template_network)

        return template_network

    def attach_interface_nodes_to_template_network_at_bottom(self,
                                                             interface_switch):
        """
        :param interface_switch: network with interface switch as root
        and interface nodes as its children
        :return: full template network with interface nodes
        connected at bottom for next time step
        with root as interface switch
        """

        latent_interface_list = []
        interface_num = 0
        for _ in interface_switch.children:
            # create a latent interface node for each interface node
            logging.debug(f'interface_switch.scope {interface_switch.scope}')

            num_features_in_one_time_step = int(
                len(self.two_time_step_params.feature_names_two_time_steps) / 2)
            interface_idx = interface_num + num_features_in_one_time_step
            interface_scope = interface_switch.scope[-1] + 1

            latent_interface_node = LatentInterface(interface_idx=interface_idx,
                                                    scope=interface_scope)
            latent_interface_list.append(latent_interface_node)
            interface_num += 1

        # perform bfs

        seen, queue = set([interface_switch]), collections.deque(
            [interface_switch])

        while queue:
            node = queue.popleft()

            if not isinstance(node, Leaf):

                # get node at level before leaf nodes.
                # This is connected to next time step interface nodes
                if all(isinstance(child, Leaf) for child in node.children):

                    # since we never add these in queue
                    # for leaf_node in node.children:
                    #     leaf_node.scope = [leaf_node.scope[0] -
                    #     num_features_in_one_time_step]

                    initial_interface_weights = [1 / len(
                        latent_interface_list)] * len(latent_interface_list)
                    interface_sum = Sum(
                        children=copy.deepcopy(latent_interface_list),
                        weights=initial_interface_weights
                    )

                    if isinstance(node, Product):
                        node.children.append(interface_sum)

                    else:
                        # children_prod_nodes = []
                        for i, child in enumerate(node.children):
                            # node.children.append(interface_sum)
                            # children = copy.deepcopy(node.children)
                            prod_interface = Product(
                                children=[child, copy.deepcopy(interface_sum)]
                            )
                            # children_prod_nodes.append(prod_interface)
                            node.children[i] = prod_interface
                        # for child in children_prod_nodes:
                        #     node.children.append(child)

                else:
                    for child in node.children:
                        if child not in seen:
                            seen.add(child)
                            queue.append(child)

            # else:
            #     logging.debug(f'leaf node type at interface layer
            #     {type(node)}')
            #     logging.debug(f'leaf node type scope at interface layer
            #     {node.scope}')
            #     # if it is leaf node,
            #     change scope to correspond to number of
            #     variables in a single time step
            #     node.scope = [node.scope[0] - num_features_in_one_time_step]
            #     logging.debug(f'leaf node type modified
            #     scope at interface layer {node.scope}')

        interface_switch = self.change_time_step_1_scopes_to_template_scopes(
            interface_switch)
        assign_ids(interface_switch)
        rebuild_scopes_bottom_up(interface_switch)
        logging.debug(f'interface_switch.scope {interface_switch.scope}')
        return interface_switch

    def change_time_step_1_scopes_to_template_scopes(self, interface_switch):

        logging.debug(
            f'in method change_time_step_1_scopes_to_template_scopes()')

        num_features_in_one_time_step = int(
            len(self.two_time_step_params.feature_names_two_time_steps) / 2)

        seen, queue = set([interface_switch]), collections.deque(
            [interface_switch])

        while queue:
            node = queue.popleft()

            if not isinstance(node, Leaf):

                for child in node.children:
                    if child not in seen:
                        seen.add(child)
                        queue.append(child)

                if isinstance(node, Max):
                    node.dec_idx = node.dec_idx - num_features_in_one_time_step

            else:
                logging.debug(f'leaf node type at interface layer {type(node)}')
                logging.debug(
                    f'leaf node type scope at interface layer {node.scope}')

                # if it is leaf node, change scope to correspond to
                # number of variables in a single time step
                node.scope = [node.scope[0] - num_features_in_one_time_step]

                logging.debug(
                    f'leaf node type modified scope at interface layer '
                    f'{node.scope}')

        assign_ids(interface_switch)
        rebuild_scopes_bottom_up(interface_switch)
        logging.debug(f'interface_switch.scope {interface_switch.scope}')

        return interface_switch

    def learn_spmn_two_time_steps(self, data):
        """
        :param data: data for two time steps, Type: numpy ndarray
        :return: learned spmn for two time steps
        """

        spmn = SPMN(self.two_time_step_params.partial_order_two_time_steps,
                    self.two_time_step_params.decision_nodes_two_time_steps,
                    self.two_time_step_params.utility_nodes_two_time_steps,
                    self.two_time_step_params.feature_names_two_time_steps,
                    self.two_time_step_params.meta_types_two_time_steps,
                    cluster_by_curr_information_set=True,
                    util_to_bin=False)

        spmn_structure_two_time_steps = spmn.learn_spmn(data)

        return spmn_structure_two_time_steps

    def wrap_sequence_into_two_time_steps(self, data,
                                          total_num_of_time_steps_varies=False):
        """"""

        if total_num_of_time_steps_varies:
            assert type(data) is list, 'When sequence length varies, data is ' \
                                       'a list of numpy arrays'
            # print("Evaluating rspn and collecting nodes to update")

            for row in range(len(data)):
                each_data_point = data[row].reshape(1, -1)
                # print("length of sequence:", self.get_len_sequence())
                two_time_step_data_row = self.get_two_time_step_data(each_data_point)
                if row == 0:
                    two_time_step_data = two_time_step_data_row
                else:
                    two_time_step_data = np.vstack(
                        (two_time_step_data, two_time_step_data_row)
                    )

        else:
            assert type(data) is np.ndarray, 'data should be of type numpy' \
                                             ' array'

            two_time_step_data = self.get_two_time_step_data(data)

        return two_time_step_data

    def get_two_time_step_data(self, data):

        len_of_sequence = int(
            data.shape[1] /
            self.two_time_step_params.length_of_each_time_slice
        )

        length_of_each_time_slice = self.two_time_step_params.length_of_each_time_slice
        print(
            f'self.two_time_step_params.length_of_each_time_slice {self.two_time_step_params.length_of_each_time_slice}')
        print(f'data.shape[1] {data.shape[1]}')
        print(f'len_of_seq {len_of_sequence}')

        for time_step in range(len_of_sequence - 1):
            if time_step == 0:
                two_time_step_data = \
                    data[:,
                    (time_step * length_of_each_time_slice):
                    (time_step + 2) * length_of_each_time_slice]
                print(f'initial {two_time_step_data.shape}')
            else:
                two_time_step_data = np.vstack(
                    (two_time_step_data,
                     data[:, (time_step * length_of_each_time_slice):
                             (time_step + 2) * length_of_each_time_slice])
                )
                print(f'timestep {time_step} {two_time_step_data.shape}')

        return two_time_step_data


class RSPMNTwoTimeStepParams:

    def __init__(self, rspmn_params):
        self.partial_order_two_time_steps = get_partial_order_two_time_steps(
            rspmn_params.partial_order)
        self.decision_nodes_two_time_steps = get_nodes_two_time_steps(
            rspmn_params.decision_nodes)
        self.utility_nodes_two_time_steps = get_nodes_two_time_steps(
            rspmn_params.utility_nodes)
        self.feature_names_two_time_steps = get_feature_names_two_time_steps(
            self.partial_order_two_time_steps)
        self.meta_types_two_time_steps = get_meta_types_two_time_steps(
            rspmn_params.meta_types
        )
        self.length_of_each_time_slice = rspmn_params.length_of_time_slice
