import numpy as np
import networkx as nx
import tools
import TIM
import time

start_time = time.time()

k = 50
data_set = "wiki_vote"
result_folder = "../results/graph/"+data_set+"/"

g = tools.load_graph(data_set)
theta = TIM.calculate_theta(g, k)

g = tools.load_graph(data_set)
cover_graph = TIM.create_rr_sets_graph(g, theta)

end_time = time.time()

nx.write_gpickle(cover_graph, result_folder+"rr_sets_graph")

file = open(result_folder+"result.txt", "w")
file.write("data set: " + data_set + "\n")
file.write("k: " + str(k) + "\n")
file.write("theta: " + str(theta) + "\n")
file.write("run time: " + str(end_time-start_time) + "\n")
file.close()
