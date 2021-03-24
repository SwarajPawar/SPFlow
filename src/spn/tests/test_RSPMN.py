# import sys
# sys.path.append('/home/hari/Desktop/Projects/Thesis_Project/SPFlow_clone/SPFlow/src')
# print(f'sys.path {sys.path}')
import unittest
import logging
from spn.algorithms.RSPMN import RSPMN
from spn.algorithms.RSPMNHelper import get_partial_order_two_time_steps, get_feature_names_two_time_steps,\
    get_nodes_two_time_steps

logging.basicConfig(level=logging.DEBUG)
import numpy as np

class TestRSPMN(unittest.TestCase):

    def setUp(self):
        feature_names = ['X0', 'X1', 'X2', 'D0', 'X3', 'X4', 'X5', 'D1', 'X6', 'X7', 'U']
        partial_order = [['X0', 'X1', 'X2'], ['D0'], ['X3', 'X4', 'X5'], ['D1'], ['X6', 'X7', 'U']]
        decision_nodes = ['D0', 'D1']
        utility_nodes = ['U', 'X5']
        util_to_bin = False

        self.rspmn = RSPMN(partial_order, decision_nodes, utility_nodes, feature_names, util_to_bin)
    #
    #     x012_data = np.arange(30).reshape(10, 3)
    #     d0_data = np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]).reshape(10, 1)
    #     x345_data = np.arange(30, 60).reshape(10, 3)
    #     d1_data = np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]).reshape(10, 1)
    #     x6u_data = np.arange(60, 80).reshape(10, 2)
    #
    #     self.data = np.concatenate((x012_data, d0_data, x345_data, d1_data, x6u_data), axis=1)
    #

    def test_build_initial_template(self):

        spmn = 'spmn'
        # num_of_variables = 3

        slice_data = self.rspmn.build_initial_template(spmn)
        logging.debug(f'sliced data is {slice_data}')
        # self.assertEqual((10, 4), slice_data.shape, msg=f'sliced data shape should be (10, 4),'
        #                                                 f' instead it is {slice_data.shape}')

    def test_get_partial_order_two_time_steps(self):

        partial_order_two_time_steps = get_partial_order_two_time_steps(self.rspmn.params.partial_order)

        req_partial_order_two_time_steps = [['X0T0', 'X1T0', 'x2T0'], ['D0T0'],
                                            ['X3T0', 'X4T0', 'X5T0'], ['D1T0'], ['X6T0', 'X7T0', 'UT0'],
                                            ['X0T1', 'X1T1', 'x2T1'], ['D0T1'],
                                            ['X3T1', 'X4T1', 'X5T1'], ['D1T1'], ['X6T1', 'X7T1', 'UT1']]

        self.assertListEqual(req_partial_order_two_time_steps, partial_order_two_time_steps)

    def test_get_feature_names_two_time_steps(self):

        partial_order_two_time_steps = [['X0T0', 'X1T0', 'x2T0'], ['D0T0'],
                                        ['X3T0', 'X4T0', 'X5T0'], ['D1T0'], ['X6T0', 'X7T0', 'UT0'],
                                        ['X0T1', 'X1T1', 'x2T1'], ['D0T1'],
                                        ['X3T1', 'X4T1', 'X5T1'], ['D1T1'], ['X6T1', 'X7T1', 'UT1']]

        feature_names_two_time_steps = get_feature_names_two_time_steps(partial_order_two_time_steps)

        req_feature_names_two_time_steps = ['X0T0', 'X1T0', 'x2T0', 'D0T0',
                                            'X3T0', 'X4T0', 'X5T0', 'D1T0', 'X6T0', 'X7T0', 'UT0',
                                            'X0T1', 'X1T1', 'x2T1', 'D0T1',
                                            'X3T1', 'X4T1', 'X5T1', 'D1T1', 'X6T1', 'X7T1', 'UT1']

        self.assertListEqual(req_feature_names_two_time_steps, feature_names_two_time_steps)

    def test_get_nodes_two_time_steps(self):

        # test decision nodes
        decision_nodes_two_time_steps = get_nodes_two_time_steps(self.rspmn.params.decision_nodes)
        req_decision_nodes_two_time_steps = ['D0T0', 'D1T0', 'D0T1', 'D1T1']
        self.assertListEqual(req_decision_nodes_two_time_steps, decision_nodes_two_time_steps)

        # test utility nodes
        utility_nodes_two_time_steps = get_nodes_two_time_steps(self.rspmn.params.utility_nodes)
        req_utility_nodes_two_time_steps = ['UT0', 'X5T0', 'UT1', 'X5T1']
        self.assertListEqual(req_utility_nodes_two_time_steps, utility_nodes_two_time_steps)

    def test_eval_rspmn_bottom_up(self, data):

        data = np.range(0, 330).reshape(-1, 33)

        # self.rspmn.eval_rspmn_bottom_up(data)

    def test_wrap_sequence_into_two_time_steps(self):

        data = np.arange(0, 550).reshape(-1, 55)
        print(data)
        two_time_step_data = \
            self.rspmn.InitialTemplate.wrap_sequence_into_two_time_steps(data)

        print(f'two_time_step_data {two_time_step_data}')



if __name__ == '__main__':
    unittest.main()
