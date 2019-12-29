import numpy as np
import tools
import competition_actions as ca

k = 30
data_set = "p2p"
# max_random_value = k
max_random_value = 3
repeat = 0

result_folder = "../results/evaluations_framework/benchmark_random"+str(max_random_value)+"/"

if max_random_value == k:
    result_folder = "../results/evaluations_framework/benchmark_random/"    


file = open(result_folder+"k"+str(k)+data_set+"_"+str(repeat)+".txt", "w")
iterations_number = 1000

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


str_print("Benchmarking against Random")
str_print("dataset: "+data_set)
str_print("k: "+str(k))
str_print("max_random_value: "+str(max_random_value))
str_print("number of iterations: "+str(iterations_number))
str_print("repeat: "+str(repeat))
str_print("\n")

v_each_seed = np.loadtxt("../results/motivations/" + data_set + "/k40seed1.txt")

nash_seed = []
nash_nash = []
nash_actions = []
for i in range(3):
    n = k + i * 5
    nash_seed.append(np.loadtxt("../results/framework/"+data_set+"/seeds/n"+str(n)+"_seeds.txt"))
    nash_nash.append(np.load("../results/framework/"+data_set+"/nash_k"+str(k)+"n"+str(n)+"/n1.npy"))
    nash_actions.append(np.load("../results/framework/"+data_set+"/nash_k"+str(k)+"n"+str(n)+"/a1.npy"))

result_nash = np.zeros((3, iterations_number))
result_veach = np.zeros((2, iterations_number))


for iter_i in range(iterations_number):
    if iter_i % 100 == 0:
        print(iter_i)
    # random action
    a2 = ca.action_random(k, v_each_seed, max_random_value)

    # nash
    for n_i in range(3):
        a1 = ca.action_nash(nash_nash[n_i], nash_actions[n_i],nash_seed[n_i])
        result_nash[n_i, iter_i] = competition(a1, a2)

    # v_each
    for v_i in range(2):
        a1 = ca.action_v_each(k, v_i+1, v_each_seed)
        result_veach[v_i, iter_i] = competition(a1, a2)


str_print("nash result")
for n_i in range(3):
    n = k + n_i * 5
    r = result_nash[n_i, :]
    str_print(" n:" + str(n))
    str_print("   diffusion-r :   mean: " + str(np.average(r)) + "   std:  " + str(np.std(r)) + "   var:  " + str(np.var(r)))
    str_print("   number of wins: "+str(np.sum(r>0)))

str_print("\n")
str_print("v_each result")
for v_i in range(2):
    r = result_veach[v_i, :]
    str_print(" v:" + str(v_i+1))
    str_print("   diffusion-r :   mean: " + str(np.average(r)) + "   std:  " + str(np.std(r)) + "   var:  " + str(np.var(r)))
    str_print("   number of wins: "+str(np.sum(r>0)))


file.close()
