"""
Created on March 30, 2018

@author: Alejandro Molina
"""

import numpy as np

from spn.algorithms.StructureLearning import get_next_operation, learn_structure
from spn.algorithms.CnetStructureLearning import get_next_operation_cnet, learn_structure_cnet
from spn.algorithms.Validity import is_valid
from spn.algorithms.AnytimeLearning import AnytimeSPN

from spn.structure.Base import Sum, assign_ids

from spn.structure.leaves.histogram.Histograms import create_histogram_leaf
from spn.structure.leaves.parametric.Parametric import create_parametric_leaf
from spn.structure.leaves.piecewise.PiecewiseLinear import create_piecewise_leaf
from spn.structure.leaves.cltree.CLTree import create_cltree_leaf
from spn.algorithms.splitting.Conditioning import (
    get_split_rows_naive_mle_conditioning,
    get_split_rows_random_conditioning,
)
import logging

logger = logging.getLogger(__name__)





def get_splitting_functions(cols, rows, ohe, threshold, rand_gen, n_jobs, k=2):
    from spn.algorithms.splitting.Clustering import get_split_rows_KMeans, get_split_rows_TSNE, get_split_rows_GMM
    from spn.algorithms.splitting.PoissonStabilityTest import get_split_cols_poisson_py
    from spn.algorithms.splitting.RDC import get_split_cols_RDC_py, get_split_rows_RDC_py

    if isinstance(cols, str):
        if cols == "rdc":
            split_cols = get_split_cols_RDC_py(threshold, rand_gen=rand_gen, ohe=ohe, n_jobs=n_jobs)
        elif cols == "poisson":
            split_cols = get_split_cols_poisson_py(threshold, n_jobs=n_jobs)
        else:
            raise AssertionError("unknown columns splitting strategy type %s" % str(cols))
    else:
        split_cols = cols

    if isinstance(rows, str):
        if rows == "rdc":
            split_rows = get_split_rows_RDC_py(rand_gen=rand_gen, ohe=ohe, n_jobs=n_jobs)
        elif rows == "kmeans":
            split_rows = get_split_rows_KMeans(n_clusters=k)
        elif rows == "tsne":
            split_rows = get_split_rows_TSNE()
        elif rows == "gmm":
            split_rows = get_split_rows_GMM()
        else:
            raise AssertionError("unknown rows splitting strategy type %s" % str(rows))
    else:
        split_rows = rows
    return split_cols, split_rows

import pandas as pd
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType

cols="rdc"
rows="kmeans"
min_instances_slice=200
threshold=0.3
ohe=False
leaves = create_histogram_leaf
rand_gen=None
cpus=-1

df = pd.read_csv("spn/data/binary/nltcs.ts.data", sep=',')
data = df.values
print(data.shape)
samples, var = data.shape
ds_context = Context(meta_types=[MetaType.DISCRETE]*var)
ds_context.add_domains(data)

split_rows = get_split_rows_XMeans()
split_cols = get_split_cols_single_RDC(rand_gen=rand_gen, ohe=ohe, n_jobs=cpus)
nextop = get_next_operation(min_instances_slice)

aspn = AnytimeSPN(split_rows, split_cols, leaves, nextop)
spn = apsn.learn_structure(data, ds_context)

from spn.io.Graphics import plot_spn

plot_spn(spn, 'basicspn.png')


import numpy as np
test_data = np.array([1]*var).reshape(-1, var)

from spn.algorithms.Inference import log_likelihood

ll = log_likelihood(spn, test_data)
print(ll, np.exp(ll))































