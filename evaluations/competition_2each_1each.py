import numpy as np
import tools
import competition_actions as ca

k = 30

result_folder = "../results/evaluations_framework/two_each_one_each/"
data_set_list = ["ca_hepth", "facebook", "p2p", "wiki_vote"]
file = open(result_folder+"k"+str(k)+".txt", "w")


def str_print(s):
    print(s)
    file.write(s+"\n")
    return


def get_seeds(a1, a2):
    seeds = set(a1.keys()).union(set(a2.keys()))
    s1 = []
    s2 = []
    for s in seeds:
        if a1.get(s, 0) > a2.get(s, 0):
            s1.append(s)
        elif a2.get(s, 0) > a1.get(s, 0):
            s2.append(s)
        elif a1.get(s, 0) == a2.get(s, 0) and a1.get(s, 0) > 0:
            if np.random.random() < 0.5:
                s1.append(s)
            else:
                s2.append(s)

    return s1, s2


def competition(data_set, nodes):
    iterations = 1000
    r = np.zeros(iterations)
    for i in range(iterations):
        a1 = ca.action_v_each(k, 2, nodes)
        a2 = ca.action_v_each(k, 1, nodes)
        s1, s2 = get_seeds(a1, a2)
        g = tools.load_graph(data_set)
        tools.diffuse(g, s1, s2)
        r[i] = len(g.graph["1"]) - len(g.graph["2"])
    return r


str_print("2-each vs 1-each")
str_print("k: "+str(k))
str_print("\n")

for data_set in data_set_list:
    str_print(data_set)
    seeds = np.loadtxt("../results/motivations/" + data_set + "/k40seed1.txt")
    r = competition(data_set, seeds)
    str_print("  diffusion-r :   mean: " + str(np.average(r)) + "   std:  " + str(np.std(r)) + "   var:  " + str(np.var(r)))
    str_print("  number of wins: " + str(sum(r > 0)))

file.close()
