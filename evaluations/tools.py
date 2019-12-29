import numpy as np
import networkx as nx


_data_sets_location_dic = {"ca_grqc": "../datasets/CA-GrQc.txt",
                           "ca_hepth": "../datasets/CA-HepTh.txt",
                           "facebook": "../datasets/facebook_combined.txt",
                           "p2p": "../datasets/p2p-Gnutella08.txt",
                           "wiki_vote": "../datasets/Wiki-Vote.txt"}

_data_sets_is_directed_dic = {"ca_grqc": False,
                              "ca_hepth": False,
                              "facebook": False,
                              "p2p": True,
                              "wiki_vote": True}

_theta_dic = {"ca_hepth": 4102607,
             "facebook": 1138509,
             "p2p": 718460,
             "wiki_vote": 3977908}


def _load_graph_directed(file):
    # loading edges from file
    edge_list = np.loadtxt(file, dtype=int)

    # create the graph
    g = nx.DiGraph()
    g.add_edges_from(edge_list)

    # adding attributes to nodes and edges
    for n in g.nodes():
        # adding weight to the edge
        in_deg = g.in_degree(n)
        if in_deg != 0:
            g.add_edges_from(g.in_edges(n), w=(1 / in_deg))

    # defining 3 lists to hold free nodes, first player and second player nodes
    g.graph['free'] = g.nodes()
    g.graph['1'] = []
    g.graph['2'] = []

    return g


def _load_graph_undirected(file):
    # loading edges from file
    edges = np.loadtxt(file, dtype=int)
    edge_list = np.zeros((2 * edges.shape[0], 2))
    edge_list[0:edges.shape[0], :] = edges[:, :]

    edge_list[edges.shape[0]:, 0] = edges[:, 1]
    edge_list[edges.shape[0]:, 1] = edges[:, 0]

    # create the graph
    g = nx.DiGraph()
    g.add_edges_from(edge_list)

    # adding attributes to nodes and edges
    for n in g.nodes():
        # adding weight to the edge
        in_deg = g.in_degree(n)
        if in_deg != 0:
            g.add_edges_from(g.in_edges(n), w=(1 / in_deg))

    # defining 3 lists to hold free nodes, first player and second player nodes
    g.graph['free'] = g.nodes()
    g.graph['1'] = []
    g.graph['2'] = []

    return g


def load_graph(name):
    if _data_sets_is_directed_dic[name]:
        g = _load_graph_directed(_data_sets_location_dic[name])
    else:
        g = _load_graph_undirected(_data_sets_location_dic[name])

    return g


def get_theta(data_set):
    return float(_theta_dic[data_set])


def _activate(g: nx.Graph, node, player):
    g.graph['free'].remove(node)
    g.graph[str(player)].append(node)
    return


def _ic_diffuse(g: nx.Graph, activated1, activated2):
    free_nodes = set(g.graph['free'])

    new_activated1 = []
    new_activated2 = []

    while activated1 + activated2:
        if np.random.randint(2) == 0:
            if activated1:
                node = activated1.pop(0)
                for edge in g.edges(node, data=True):
                    if edge[1] in free_nodes:
                        r = np.random.random()
                        if r < edge[2]['w']:
                            # node is activated
                            new_activated1.append(edge[1])
                            _activate(g, edge[1], 1)
                            free_nodes.remove(edge[1])

            if activated2:
                node = activated2.pop(0)
                for edge in g.edges(node, data=True):
                    if edge[1] in free_nodes:
                        r = np.random.random()
                        if r < edge[2]['w']:
                            # node is activated
                            new_activated2.append(edge[1])
                            _activate(g, edge[1], 2)
                            free_nodes.remove(edge[1])
        else:
            if activated2:
                node = activated2.pop(0)
                for edge in g.edges(node, data=True):
                    if edge[1] in free_nodes:
                        r = np.random.random()
                        if r < edge[2]['w']:
                            # node is activated
                            new_activated2.append(edge[1])
                            _activate(g, edge[1], 2)
                            free_nodes.remove(edge[1])

            if activated1:
                node = activated1.pop(0)
                for edge in g.edges(node, data=True):
                    if edge[1] in free_nodes:
                        r = np.random.random()
                        if r < edge[2]['w']:
                            # node is activated
                            new_activated1.append(edge[1])
                            _activate(g, edge[1], 1)
                            free_nodes.remove(edge[1])

    return new_activated1, new_activated2


def diffuse(g: nx.Graph, seed_set1, seed_set2):
    for node in seed_set1:
        _activate(g, node, 1)
    for node in seed_set2:
        _activate(g, node, 2)

    new_activated1 = list(seed_set1)
    new_activated2 = list(seed_set2)
    while len(new_activated1 + new_activated2) != 0:
        new_activated1, new_activated2 = _ic_diffuse(g, new_activated1, new_activated2)

    return
