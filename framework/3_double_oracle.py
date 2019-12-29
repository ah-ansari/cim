import numpy as np
import gambit
import time
import os
import best_resp_oracle


data_set = "p2p"
k = 30
n = 40

result_folder = "../results/framework/"+data_set+"/nash_k"+str(k)+"n"+str(n)+"/"

if not os.path.isdir(os.path.join(os.getcwd(), result_folder)):
    print("Error, the directory does not exist")
    exit(1)

file = open(result_folder+"result.txt", "w")


def str_print(s):
    print(s)
    file.write(s+"\n")
    return


def calculate_payoff(a1, a2, values):
    r1 = 0
    r2 = 0
    for i in range(len(a1)):
        if a1[i] > a2[i]:
            r1 += values[i]
        elif a2[i] > a1[i]:
            r2 += values[i]
    return r1-r2


def create_payoff_matrix(actions1, actions2, values):
    payoff = np.zeros((actions1.shape[0], actions2.shape[0]), dtype=int)

    for a1 in range(actions1.shape[0]):
        for a2 in range(actions2.shape[0]):
            payoff[a1, a2] = calculate_payoff(actions1[a1], actions2[a2], values)

    return payoff


def find_nash(payoff):
    g = gambit.Game.from_arrays(payoff, -1*payoff)
    nash = gambit.nash.lp_solve(g, rational=True)
    l = len(nash[0])/2
    nash1 = np.zeros(l)
    nash2 = np.zeros(l)
    for i in range(l):
        nash1[i] = float(nash[0][i])
        nash2[i] = float(nash[0][i+l])
    return nash1, nash2


def double_oracle(values, k):
    initial_action = np.zeros(len(values), dtype=int)
    for i in range(k):
        initial_action[i] = 1

    actions1 = np.array([initial_action])
    actions2 = np.array([initial_action])

    for iteration in range(250):
        payoff = create_payoff_matrix(actions1, actions2, values)
        nash1, nash2 = find_nash(payoff)
        r, v_r = best_resp_oracle.best_response(actions2, nash2, values, k)
        c, v_c = best_resp_oracle.best_response(actions1, nash1, values, k)

        str_print("iteration: " + str(iteration) + "   v_r:" + str(v_r) + "  v_c:" + str(v_c))
        str_print(str(r))
        str_print(str(c))
        str_print("-------------------------------------------------------------------------")

        in_actions1 = (actions1 == r).all(axis=1).nonzero()[0]
        in_actions2 = (actions2 == c).all(axis=1).nonzero()[0]
        if (len(in_actions1) > 0) and (len(in_actions2) > 0):
            str_print("the algorithm has converged in iteration: "+str(iteration)+"   term condition convergence")
            return actions1, nash1, actions2, nash2

        if (v_r + v_c) < 0.5:
            str_print("the algorithm has converged in iteration: "+str(iteration)+"   term condition eps = 0.5")
            return actions1, nash1, actions2, nash2

        actions1 = np.insert(actions1, [0], r, axis=0)
        actions2 = np.insert(actions2, [0], c, axis=0)

    str_print("the algorithm has not converged")
    return actions1, nash1, actions2, nash2


values = np.loadtxt("../results/game/"+data_set+"/seeds/n"+str(n)+"_values.txt")
t1 = time.time()

a1, n1, a2, n2 = double_oracle(values, k)

t2 = time.time()
str_print("\n"+"running time: " + str(t2 - t1))
str_print("\n")

np.save(result_folder+"a1.npy", a1)
np.save(result_folder+"n1.npy", n1)
np.save(result_folder+"a2.npy", a2)
np.save(result_folder+"n2.npy", n2)

file.close()
