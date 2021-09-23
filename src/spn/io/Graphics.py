"""
Created on March 29, 2018

@author: Alejandro Molina
"""
from matplotlib.ticker import NullLocator
from networkx.drawing.nx_agraph import graphviz_layout

# import matplotlib
# matplotlib.use('Agg')
import logging

logger = logging.getLogger(__name__)


def get_networkx_obj(spn, feature_labels=None, draw_interfaces = False):
    import networkx as nx
    from spn.structure.Base import Sum, Product, Leaf, get_nodes_by_type, Max
    from spn.structure.leaves.spmnLeaves.SPMNLeaf import Utility, State
    import numpy as np

    all_nodes = get_nodes_by_type(spn)
    logger.info(all_nodes)

    g = nx.Graph()

    labels = {}
    for n in all_nodes:
        color = "#DDDDDD"
        if isinstance(n, Sum):
            label = "+"
            shape = 'o'
        elif isinstance(n, Product):
            label = "x"
            shape = 'o'
        elif isinstance(n, Max):
            label = n.feature_name
            shape = 's'
        elif isinstance(n, Utility):
            shape = 'd'
            if feature_labels is not None:
                label = feature_labels[n.scope[0]]
            else:
                label = "U" + str(n.scope[0])
        elif isinstance(n, State):
            shape = 'o'
            if n.scope[0]==0:
                label = "S1"
                color = "#428df5"
            else:
                label = "S2"
                color = "#ff7d8c"
        else:
            if feature_labels is not None:
                label = feature_labels[n.scope[0]]
                shape = 'o'
            else:
                label = "V" + str(n.scope[0])
                shape = 'o'



        g.add_node(n.id, s=shape, c=color)
        labels[n.id] = label


        if draw_interfaces and isinstance(n, State):
            for i, c in enumerate(n.interface_links):
                g.add_edge(c.id, n.id, weight=i, style='dashed')
            continue
        elif isinstance(n, Leaf):
            continue
        for i, c in enumerate(n.children):
            edge_label = ""
            if isinstance(n, Sum):
                edge_label = np.round(n.weights[i], 2)
            g.add_edge(c.id, n.id, weight=edge_label, style='solid')

            if isinstance(n, Max):
                edge_label = np.round(n.dec_values[i], 2)
            g.add_edge(c.id, n.id, weight=edge_label, style='solid')

    return g, labels


def plot_spn(spn, fname="plot.pdf", feature_labels = None, draw_interfaces = False):

    import networkx as nx
    from networkx.drawing.nx_pydot import graphviz_layout

    import matplotlib.pyplot as plt

    plt.clf()

    g, labels = get_networkx_obj(spn, feature_labels, draw_interfaces)

    pos = graphviz_layout(g, prog="dot")
    # plt.figure(figsize=(18, 12))
    ax = plt.gca()

    # ax.invert_yaxis()

    node_shapes = set(((node_shape[1]["s"],node_shape[1]["c"]) for node_shape in g.nodes(data=True)))

    font_size = 6

    for node_shape, node_color in node_shapes:
        nx.draw(
            g,
            pos,
            with_labels=True,
            arrows=False,
            node_color=node_color,
            edge_color="#888888",
            width=1,
            node_size=360,
            labels=labels,
            font_size=font_size,
            node_shape=node_shape,
            nodelist=[
                sNode[0] for sNode in filter(lambda x: x[1]["s"] == node_shape and x[1]["c"] == node_color, g.nodes(data=True))
            ],
            edgelist=[
                edge[0] for edge in filter(lambda x: x[1]=='solid', nx.get_edge_attributes(g,'style').items())
            ]
        )

    edge_labels_dict = nx.get_edge_attributes(g, "weight")

    if draw_interfaces:
        interface_edges = [
                edge[0] for edge in filter(lambda x: x[1]=='dashed', nx.get_edge_attributes(g,'style').items())
            ]

        nx.draw_networkx_edges(
            g,
            pos,
            edgelist=interface_edges,
            style='dashed',
            edge_color='r',
            alpha=0.5,
            width=0.5
        )
        for edge in interface_edges:
            del edge_labels_dict[edge]

    ax.collections[0].set_edgecolor("#333333")
    edge_labels = nx.draw_networkx_edge_labels(
        g, pos=pos, edge_labels=edge_labels_dict, font_size=font_size, clip_on=False, alpha=0.6
    )


    xpos = list(map(lambda p: p[0], pos.values()))
    ypos = list(map(lambda p: p[1], pos.values()))

    ax.set_xlim(min(xpos) - 35, max(xpos) + 35)
    ax.set_ylim(min(ypos) - 35, max(ypos) + 35)
    plt.tight_layout()
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.savefig(fname, bbox_inches="tight", pad_inches=0, dpi=1000)


def plot_spn2(spn, fname="plot.pdf"):
    import networkx as nx
    import matplotlib.pyplot as plt

    g, _ = get_networkx_obj(spn)

    pos = graphviz_layout(g, prog="dot")
    nx.draw(g, pos, with_labels=False, arrows=False)
    plt.savefig(fname)


def plot_spn_to_svg(root_node, fname="plot.svg"):
    import networkx.drawing.nx_pydot as nxpd

    g, _ = get_networkx_obj(root_node)

    pdG = nxpd.to_pydot(g)
    svg_string = pdG.create_svg()

    f = open(fname, "wb")
    f.write(svg_string)
