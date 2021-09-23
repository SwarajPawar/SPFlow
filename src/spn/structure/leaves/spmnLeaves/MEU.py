from spn.structure.leaves.histogram.MPE import histogram_mode, \
    histogram_top_down, histogram_bottom_up_ll
from spn.algorithms.MPE import get_mpe_top_down_leaf

from spn.algorithms.MPE import add_node_mpe

from spn.structure.leaves.spmnLeaves.SPMNLeaf import Utility
import numpy as np

from spn.structure.leaves.spmnLeaves.Inference import utility_value

from spn.structure.leaves.spmnLeaves.Inference import latent_interface_likelihood

from spn.structure.leaves.spmnLeaves.SPMNLeaf import LatentInterface


def utility_mode(node):
    return histogram_mode(node)


def utility_bottom_up_uVal(node, data=None, dtype=np.float64):
    uVal = utility_value(node, data=data, dtype=dtype)
    mpe_ids = np.isnan(data[:, node.scope[0]])
    mode_data = np.ones((1, data.shape[1])) * histogram_mode(node)
    uVal[mpe_ids] = utility_value(node, data=mode_data, dtype=dtype)

    return uVal


def utility_top_down(node, input_vals, lls_per_node, data=None):
    get_mpe_top_down_leaf(node, input_vals, data=data, mode=utility_mode(node))


def add_utility_mpe_support():
    add_node_mpe(Utility, histogram_bottom_up_ll, histogram_top_down)


def latent_interface_mode(node):
    assert True, f'latent interface node data does not have mode or mpe value. ' \
        f'data must contain value from corresponding bottom time step interface node'
    return node.interface_idx


def latent_interface_bottom_up_Val(node, data=None, dtype=np.float64):

    inference_val = latent_interface_likelihood(node, data=data, dtype=dtype)

    mpe_ids = np.isnan(data[:, node.interface_idx])
    n_mpe = np.sum(mpe_ids)
    # assert n_mpe == node.interface_idx, f'latent interface node data does not have mpe value. ' \
    #     f'data must contain value from corresponding bottom time step interface node'

    return inference_val


def latent_interface_top_down(node, input_vals, lls_per_node, data=None):
    get_mpe_top_down_leaf(node, input_vals, data=data, mode=latent_interface_mode(node))


def add_latent_interface_mpe_support():
    add_node_mpe(LatentInterface, latent_interface_bottom_up_Val, latent_interface_top_down)