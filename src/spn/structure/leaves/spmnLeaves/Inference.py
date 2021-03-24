
import numpy as np

from spn.algorithms.Inference import EPSILON, add_node_likelihood

from spn.structure.leaves.spmnLeaves.SPMNLeaf import Utility
from spn.structure.leaves.histogram.Inference import histogram_likelihood


import logging

from spn.structure.leaves.spmnLeaves.SPMNLeaf import LatentInterface


def utility_value(node, data=None, dtype=np.float64):
    uVal = np.ones((data.shape[0], 1), dtype=dtype)

    logging.debug(f'utility scope {node.scope[0]}')
    nd = data[:, node.scope[0]]
    marg_ids = np.isnan(nd)

    uVal[~marg_ids] = nd[~marg_ids].reshape((-1,1))

    uVal[uVal < EPSILON] = EPSILON

    return uVal


def add_utility_inference_support():
    add_node_likelihood(Utility, histogram_likelihood)


def latent_interface_likelihood(node, data=None, dtype=np.float64):
    logging.debug(f' in function  latent_interface_likelihood')
    probs = np.ones((data.shape[0], 1), dtype=dtype)

    # logging.debug(f' type(node) { type(node)}')
    #
    # logging.debug(f'node.scope {node.scope}')
    logging.debug(f'node.interface_idx {node.interface_idx}')
    logging.debug(f'data {data}')

    node_data = data[:, node.interface_idx]
    marg_ids = np.isnan(node_data)

    # latent interface likelihood is the corresponding bottom time step interface node's likelihood
    # this is added to data from previous time step's log likelihood on the template network
    probs[~marg_ids] = np.exp(node_data).reshape(-1, 1)

    return probs


def add_latent_interface_inference_support():
    add_node_likelihood(LatentInterface, latent_interface_likelihood)