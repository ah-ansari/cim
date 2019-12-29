import numpy as np
import tools
import time

data_set = "wiki_vote"
result_folder = "../results/motivations/"+data_set+"/"

k = 10
seed_set1 = np.loadtxt(result_folder+"k"+str(k)+"seed1.txt")
seed_set2 = np.loadtxt(result_folder+"k"+str(k)+"seed2.txt")

iterations = 5000
r1 = np.zeros(iterations)
r2 = np.zeros(iterations)
r = np.zeros(iterations)

start_time = time.time()

for i in range(iterations):
    g = tools.load_graph(data_set)
    tools.diffuse(g, seed_set1, seed_set2)

    n1 = len(g.graph["1"])
    n2 = len(g.graph["2"])
    r1[i] = n1
    r2[i] = n2
    r[i] = n1 - n2

end_time = time.time()

file = open(result_folder+"result_diffusion.txt", "a")
file.write("k: " + str(k) + "\n")
file.write("run time: " + str(end_time-start_time) + "\n")
file.write("seed1: " + str(seed_set1) + "\n")
file.write("seed2: " + str(seed_set2) + "\n")
file.write("diffusion-r1    mean: "+str(np.average(r1))+"   std:  "+str(np.std(r1))+"   var:  "+str(np.var(r1))+"\n")
file.write("diffusion-r2    mean: "+str(np.average(r2))+"   std:  "+str(np.std(r2))+"   var:  "+str(np.var(r2))+"\n")
file.write("diffusion-r     mean: "+str(np.average(r))+"   std:  "+str(np.std(r))+"   var:  "+str(np.var(r))+"\n")
file.write("\n")
file.close()
