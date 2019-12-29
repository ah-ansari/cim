import numpy as np
import networkx as nx
import scipy
import tools


data_set = "wiki_vote"
result_folder = "../results/evaluations_payoff/"+data_set+"/"

random_allocations = np.loadtxt("random_allocations.txt")
seed_set = np.loadtxt("../results/framework/"+data_set+"/seeds/n50_seeds.txt")
our_method_values = np.loadtxt("../results/framework/"+data_set+"/seeds/n50_values.txt")


def get_seed_sets(seed_set, random_allocation):
    s1 = []
    s2 = []
    for i in range(len(random_allocation)):
        if random_allocation[i] == 1:
            s1.append(seed_set[i])
        elif random_allocation[i] == 2:
            s2.append(seed_set[i])
    return s1, s2


def diffusion_payoff(s1, s2):
    iterations = 5000
    r = np.zeros(iterations)
    for i in range(iterations):
        g = tools.load_graph(data_set)
        tools.diffuse(g, s1, s2)
        r[i] = len(g.graph["1"]) - len(g.graph["2"])
    return r


def our_method_payoff(values, random_allocation):
    r1 = 0
    r2 = 0
    for i in range(len(random_allocation)):
        if random_allocation[i] == 1:
            r1 += values[i]
        elif random_allocation[i] == 2:
            r2 += values[i]
    return r1-r2


def basic_rr_set_payoff(s1, s2):
    rr_sets_graph = nx.read_gpickle("../results/graph/"+data_set+"/rr_sets_graph")
    r1 = 0
    r2 = 0
    for s in s1:
        r1 += rr_sets_graph.degree(s)
    for s in s2:
        r2 += rr_sets_graph.degree(s)

    return r1-r2


def calculate_payoff_by_value(s1, s2, values):
    r1 = 0
    r2 = 0
    for s in s1:
        r1 += values[s]
    for s in s2:
        r2 += values[s]

    return r1-r2


def influence_matrix_get_F(g):
    ag = nx.to_numpy_matrix(g, weight='w')

    if not tools._data_sets_is_directed_dic[data_set]:
        eig = scipy.linalg.eigvalsh(ag)[-1]
    else:
        eig = scipy.linalg.eigvals(ag)
        eig = max(eig)
        eig = eig.real

    F = np.identity(ag.shape[0])-(0.6/eig)*ag
    F = np.linalg.inv(F)
    return F


def calculate_payoff_influence_matrix(seed1, seed2, F):
    threshold = 0.1
    a1 = [g.nodes().index(i) for i in seed1]
    a2 = [g.nodes().index(i) for i in seed2]

    A1 = F[a1, :]
    A2 = F[a2, :]

    A1 = np.sum(A1, axis=0)
    A2 = np.sum(A2, axis=0)
    diff = A1 - A2
    r = np.sum(diff > threshold) - np.sum(diff < -1 * threshold)
    return r


file = open(result_folder + "payoff_evaluation_result.txt", "w")

g = tools.load_graph(data_set)
degree_centrality_values = nx.degree_centrality(g)
betweenness_centrality_values = nx.betweenness_centrality(g, normalized=True, weight='w')
closeness_centrality_values = nx.closeness_centrality(g, normalized=True)
eigenvector_centrality_values = nx.eigenvector_centrality(g, weight='w')
katz_centrality_values = nx.katz_centrality(g, weight='w', normalized=False)

if g.selfloop_edges():
    file.write("contains self loop edges\n\n")
    g_c = g.copy()
    g_c.remove_edges_from(g_c.selfloop_edges())
    corenumber_centrality_values = nx.core_number(g_c)
else:
    corenumber_centrality_values = nx.core_number(g)

pagerank_values = nx.pagerank(g, weight='w')
influence_matrix_F = influence_matrix_get_F(g)

file.write("data_set:" + data_set + "\n\n")

print(data_set)
for allocation_index in range(random_allocations.shape[0]):
    print(allocation_index)
    file.write("test:"+str(allocation_index)+"  ------------------------\n")

    s1, s2 = get_seed_sets(seed_set, random_allocations[allocation_index])

    # payoff by diffusion
    r = diffusion_payoff(s1, s2)
    file.write(" diffusion-r :   mean: "+str(np.average(r))+"   std:  "+str(np.std(r))+"   var:  "+str(np.var(r))+"\n")

    # our method payoff
    r = our_method_payoff(our_method_values, random_allocations[allocation_index])
    file.write(" our method payoff :  " + str(r) + "\n")

    # non-competitive RR set based payoff
    r = basic_rr_set_payoff(s1, s2)
    r = (g.number_of_nodes()/tools.get_theta(data_set)) * r
    file.write(" non-competitive method :  " + str(r) + "\n")

    # influence matrix
    r = calculate_payoff_influence_matrix(s1, s2, influence_matrix_F)
    file.write(" influence_matrix :  " + str(r) + "\n")

    # degree
    r = calculate_payoff_by_value(s1, s2, degree_centrality_values)
    file.write(" degree :  " + str(r) + "\n")

    # betweenness
    r = calculate_payoff_by_value(s1, s2, betweenness_centrality_values)
    file.write(" betweenness :  " + str(r) + "\n")

    # closeness
    r = calculate_payoff_by_value(s1, s2, closeness_centrality_values)
    file.write(" closeness :  " + str(r) + "\n")

    # eigenvector
    r = calculate_payoff_by_value(s1, s2, eigenvector_centrality_values)
    file.write(" eigenvector :  " + str(r) + "\n")

    # katz
    r = calculate_payoff_by_value(s1, s2, katz_centrality_values)
    file.write(" katz :  " + str(r) + "\n")

    # core number
    r = calculate_payoff_by_value(s1, s2, corenumber_centrality_values)
    file.write(" core number :  " + str(r) + "\n")

    # pagerank
    r = calculate_payoff_by_value(s1, s2, pagerank_values)
    file.write(" pagerank :  " + str(r) + "\n")

file.close()
