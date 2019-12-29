import numpy as np
import networkx as nx
import tools
import TIM
import time


k = 10
data_set = "wiki_vote"
result_folder = "../results/motivations/"+data_set+"/"

cover_graph = nx.read_gpickle("../results/graph/"+data_set+"/rr_sets_graph")

start_time = time.time()

g = tools.load_graph(data_set)
seed_set = TIM.seed_selection(list(g.nodes()), k, cover_graph)

end_time = time.time()

np.savetxt(result_folder+"k"+str(k)+"seed1.txt", seed_set)

file = open(result_folder+"result_seed1.txt", "a")
file.write("k: " + str(k) + "\n")
file.write("run time: " + str(end_time-start_time) + "\n")
file.write("seed1: " + str(seed_set) + "\n")
file.write("\n")
file.close()
