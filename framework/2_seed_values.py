import numpy as np
import networkx as nx
import operator
import time
import tools


data_set = "wiki_vote"
result_folder = "../results/framework/" + data_set + "/seeds/"


def seed_selection(free_nodes, n, cover_graph):
    seed_set = []
    for i in range(n):
        degrees = cover_graph.degree(free_nodes)
        max_node = max(degrees.items(), key=operator.itemgetter(1))[0]
        seed_set.append(max_node)
        free_nodes.remove(max_node)
        for edge in cover_graph.edges(max_node):
            cover_graph.remove_node(edge[1])

    return seed_set


n_list = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]

run_time = np.zeros(len(n_list))

for n in n_list:
    print("n: " + str(n))
    start_time = time.time()
    g = tools.load_graph(data_set)
    n_nodes = g.number_of_nodes()

    rr_sets_graph = nx.read_gpickle("../results/graph/"+data_set+"/rr_sets_graph")
    seed_set = seed_selection(list(g.nodes()), n, rr_sets_graph)

    rr_sets_graph = nx.read_gpickle("../results/graph/"+data_set+"/rr_sets_graph")
    value_graph = nx.DiGraph()
    for seed in seed_set:
        for edge in rr_sets_graph.edges(seed):
            value_graph.add_edge(edge[0], edge[1])

    seed_set = set(seed_set)
    for node in value_graph.nodes():
        if node not in seed_set:
            in_deg = value_graph.in_degree(node)
            value_graph.add_edges_from(value_graph.in_edges(node), w=(1 / in_deg))

    seed_dict = {}
    for seed in seed_set:
        seed_dict[seed] = (float(n_nodes)/float(tools.get_theta(data_set))) * value_graph.degree(seed, weight="w")

    sorted_seed = sorted(seed_dict.items(), key=operator.itemgetter(1), reverse=True)
    sorted_seed = np.array(sorted_seed)

    run_time[n_list.index(n)] = time.time() - start_time
    np.savetxt(result_folder + "n"+str(n)+"_seeds.txt", sorted_seed[:, 0])
    np.savetxt(result_folder + "n"+str(n)+"_values.txt", sorted_seed[:, 1])

file = open(result_folder+"result.txt", "w")
file.write("data set: " + data_set + "\n")
file.write("n_list: \n")
file.write(str(n_list))
file.write("\n")
file.write("run time: \n")
file.write(str(run_time))
file.write("\n")
file.close()
