import collections
import logging

from spn.algorithms.EM import get_node_updates_for_EM
from spn.algorithms.Gradient import get_node_gradients
from spn.algorithms.RSPMN.TemplateUtil import gradient_backward
from spn.structure.Base import get_nodes_by_type


def soft_em(self, template, data, iterations=5, **kwargs):


    node_updates = get_node_updates_for_EM()

    # lls_per_node = np.zeros((data.shape[0], get_number_of_nodes(template)))

    for _ in range(iterations):
        # one pass bottom up evaluating the likelihoods
        ll, unrolled_network_lls_per_node = self.log_likelihood(template, data)

        template = self.rspmn_gradient_backward(template, data,
                                                unrolled_network_lls_per_node,
                                                node_updates, **kwargs)

        self.template = template

        return template


def rspmn_gradient_backward(self, template, data,
                            unrolled_network_lls_per_node, node_updates,
                            **kwargs):

    num_variables_each_time_step, total_num_of_time_steps, \
        initial_num_latent_interface_nodes = \
        self.get_params_for_get_each_time_step_data_for_template(template,
                                                                 data)

    latent_queue = collections.deque()
    for time_step_num in range(total_num_of_time_steps -1):
        lls_per_node = unrolled_network_lls_per_node[
            total_num_of_time_steps - time_step_num
            ]

        each_time_step_data_for_template = \
            self.get_each_time_step_data_for_template(
                data, time_step_num,
                total_num_of_time_steps,
                lls_per_node,
                initial_num_latent_interface_nodes,
                num_variables_each_time_step,
                bottom_up=False
            )
        node_gradients = get_node_gradients()[0].copy()

        if time_step_num == 0:

            R = lls_per_node[:, 0]
            gradients, latent_interface_dict = \
                gradient_backward(self.InitialTemplate.top_network,
                                  lls_per_node, node_gradients,
                                  data=each_time_step_data_for_template)

            next_time_step_latent_queue = collections.deque()
            top_down_pass_val_dict = {}
            i = 0
            for latent_interface_node, top_down_pass_val in \
                    latent_interface_dict.items():

                template_child_num = i % len(template.children)
                template_child = template.children[template_child_num]
                top_down_pass_val_dict[template_child] = \
                    top_down_pass_val

                if (i + 1) % len(template.children) == 0:
                    next_time_step_latent_queue.append(
                        top_down_pass_val_dict
                    )
                    top_down_pass_val_dict = {}
                i += 1

            for node_type, func in node_updates.items():
                for node in get_nodes_by_type(self.InitialTemplate.top_network,
                                              node_type):
                    func(
                        node,
                        node_lls=lls_per_node[:, node.id],
                        node_gradients=gradients[:, node.id],
                        root_lls=R,
                        all_lls=lls_per_node,
                        all_gradients=gradients,
                        data=each_time_step_data_for_template,
                        **kwargs
                    )

        else:

            R = lls_per_node[:, 0]
            next_time_step_latent_queue = collections.deque()
            while latent_queue:
                print(f'length of latent queue {len(latent_queue)}')
                template.top_down_pass_val = latent_queue.popleft()
                gradients, latent_interface_dict = \
                    gradient_backward(template,
                                      lls_per_node, node_gradients,
                                      data=each_time_step_data_for_template)

                top_down_pass_val_dict = {}
                i = 0
                for latent_interface_node, top_down_pass_val in \
                        latent_interface_dict.items():

                    template_child_num = i % len(template.children)
                    template_child = template.children[template_child_num]
                    top_down_pass_val_dict[template_child] = \
                        top_down_pass_val

                    if (i + 1) % len(template.children) == 0:
                        next_time_step_latent_queue.append(
                            top_down_pass_val_dict
                        )
                        top_down_pass_val_dict = {}
                    i += 1

                for node_type, func in node_updates.items():
                    for node in get_nodes_by_type(template, node_type):
                        func(
                            node,
                            node_lls=lls_per_node[:, node.id],
                            node_gradients=gradients[:, node.id],
                            root_lls=R,
                            all_lls=lls_per_node,
                            all_gradients=gradients,
                            data=data,
                            **kwargs
                        )
        logging.debug(f'latent_interface_dict {latent_interface_dict}')

        latent_queue = next_time_step_latent_queue

        # if self.template.interface_winner.any(np.inf):
        #     raise Exception(f'All instances are not passed to
        #     the corresponding latent interface nodes')
    return template
