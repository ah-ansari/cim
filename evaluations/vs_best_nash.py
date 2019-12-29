import numpy as np
import tools
import competition_actions as ca


k = 30
nn = 35
data_set = "wiki_vote"

result_folder = "../results/evaluations_framework/vs_best_nash/"
file = open(result_folder+data_set+"k"+str(k)+"n"+str(nn)+".txt", "w")


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


def competition(a1, a2):
    s1, s2 = get_seeds(a1, a2)
    g = tools.load_graph(data_set)
    tools.diffuse(g, s1, s2)
    return len(g.graph["1"]) - len(g.graph["2"])


best_response_seeds = np.loadtxt("../results/framework/"+data_set+"/seeds/n"+str(nn)+"_seeds.txt")
best_response_values = np.loadtxt("../results/framework/"+data_set+"/seeds/n"+str(nn)+"_values.txt")

nash_seed = []
nash_nash = []
nash_actions = []
for i in range(3):
    n = k + i * 5
    nash_seed.append(np.loadtxt("../results/framework/"+data_set+"/seeds/n"+str(n)+"_seeds.txt"))
    nash_nash.append(np.load("../results/framework/"+data_set+"/nash_k"+str(k)+"n"+str(n)+"/n1.npy"))
    nash_actions.append(np.load("../results/framework/"+data_set+"/nash_k"+str(k)+"n"+str(n)+"/a1.npy"))

iterations_number = 1000

result_nash = np.zeros((3, iterations_number))

str_print("nash Vs the best response against nash:")
str_print("dataset: "+data_set)
str_print("iterations: "+str(iterations_number))
str_print("k: "+str(k))
str_print("nn: "+str(nn))
str_print("\n")

for iter_i in range(iterations_number):
    # defeat action
    a = ca.action_nash(nash_nash[1], nash_actions[1], nash_seed[1])
    a2 = ca.best_resp(a, k, best_response_seeds, best_response_values)

    # nash
    for n_i in range(3):
        a1 = ca.action_nash(nash_nash[n_i], nash_actions[n_i], nash_seed[n_i])
        result_nash[n_i, iter_i] = competition(a1, a2)

str_print("nash result:")
for n_i in range(3):
    n = k + n_i * 5
    r = result_nash[n_i, :]
    str_print(" n:" + str(n))
    str_print("  diffusion-r :   mean: " + str(np.average(r)) + "   std:  " + str(np.std(r)) + "   var:  " + str(np.var(r)))
    str_print("  number of wins: "+str(np.sum(r>0)))

file.close()
