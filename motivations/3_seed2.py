import numpy as np
import tools
import CTIM
import time


theta_dic = {"ca_hepth": 4102607,
             "facebook": 1138509,
             "p2p": 718460,
             "wiki_vote": 3977908}


k = 10
data_set = "wiki_vote"
theta = theta_dic[data_set]
result_folder = "../results/motivations/"+data_set+"/"

seed_set1 = np.loadtxt(result_folder+"k"+str(k)+"seed1.txt")

start_time = time.time()


g = tools.load_graph(data_set)
cover_graph = CTIM.create_rr_sets_graph(g, theta, seed_set1)
seed_set2 = CTIM.seed_selection(list(g.nodes()), k, cover_graph)

end_time = time.time()

np.savetxt(result_folder+"k"+str(k)+"seed2.txt", seed_set2)

file = open(result_folder+"result_seed2.txt", "a")
file.write("k: " + str(k) + "\n")
file.write("theta: " + str(k) + "\n")
file.write("run time: " + str(end_time-start_time) + "\n")
file.write("seed2: " + str(seed_set2) + "\n")
file.write("\n")
file.close()
