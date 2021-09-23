import numpy as np
from spn.structure.StatisticalTypes import MetaType
from spn.structure.leaves.histogram.Histograms import Histogram, create_histogram_leaf

# class SPMNLeaf(Histogram):
#
#     def __init__(self, scope):
from spn.structure.Base import Leaf


def create_spmn_leaf(data, ds_context, scope):
    assert len(scope) == 1, "scope of univariate histogram for more than one variable?"
    assert data.shape[1] == 1, "data has more than one feature?"

    # data = data[~np.isnan(data)]

    idx = scope[0]
    meta_type = ds_context.meta_types[idx]

    if meta_type == MetaType.UTILITY:
        hist = create_histogram_leaf(data, ds_context, scope)
        return Utility(hist.breaks, hist.densities, hist.bin_repr_points, scope=idx)
    else:
        return create_histogram_leaf(data, ds_context, scope)


class Utility(Histogram):

    def __init__(self, breaks, densities, bin_repr_points, scope=None, type_=None, meta_type=MetaType.UTILITY):
        Leaf.__init__(self, scope=scope)
        # has same member variables as histogram
        Histogram.__init__(self, breaks, densities, bin_repr_points, scope,
                           type_=None, meta_type=MetaType.UTILITY)


class LatentInterface(Leaf):
    def __init__(self, interface_idx, bin_val=0, scope=None):
        Leaf.__init__(self, scope=scope)

        self.interface_idx = interface_idx  # num of features plus interface node num
        self.bin_val = bin_val  # 1 if data passes through corresponding interface node
        self.bottom_up_val = None  # likelihood or meu value of the corresponding interface node in the next time step
