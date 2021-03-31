

import numpy as np

import matplotlib.pyplot as plt

path = "cross_new"
datasets = ["kdd"]

upper = {"nltcs":
	{"ll": [-6.05057148300942], "n": [2152]},
        "msnbc":
	{"ll": [-6.045740974137786],	"n": [5142]},
        "plants":
	{"ll": [-13.328004042213157],		"n": [10671]},
        "kdd":
	{"ll": [-2.1610551668591897],		"n": [3963]},
        "baudio":
	{"ll": [-40.07111208519845]	,	"n": [13621]},
        "jester":
	{"ll": [-53.10895478609421]	,	"n": [16118]},
        "bnetflix":
	{"ll": [-57.05938603881737]	,	"n": [4718]}
    }

colors = ["aqua", "palegreen", "pink"]
    
for dataset in datasets:
    
    lls = list()
    nodes_k = list()
    for k in range(1,4):
        f = open(f"{path}/{dataset}/{k}/stats.txt","r")
        stats = f.readlines()
        f.close()
        ll = stats[2][stats[2].index("[")+1:-2].split(", ")
        lls.append([float(x) for x in ll])
        nodes = stats[3][stats[3].index("[")+1:-1].split(", ")
        nodes_k.append([float(x) for x in nodes])
        
    plt.close()
    total_ll = np.zeros(min([len(lls[i]) for i in range(len(lls))]))
    upperll = [upper[dataset]["ll"]] * len(total_ll)
    plt.plot(upperll, linestyle="dotted", color ="blue", label="Upper Limit")
    for i in range(len(lls)):
    	plt.plot(lls[i], marker="o", color =colors[i], label=(i+1))
    	total_ll += np.array(lls[i][:len(total_ll)])
    avg_ll = total_ll/len(lls)
    plt.plot(avg_ll, marker="o", color ="black", label="Mean")
    plt.title(f"{dataset} Log Likelihood")
    plt.legend()
    plt.savefig(f"{path}/{dataset}/ll.png", dpi=150)
    plt.close()
    
    
    total_nodes = np.zeros(min([len(nodes_k[i]) for i in range(len(nodes_k))]))
    uppern = [upper[dataset]["n"]] * len(total_nodes)
    plt.plot(uppern, linestyle="dotted", color ="blue", label="Upper Limit")
    for i in range(len(nodes_k)):
    	plt.plot(nodes_k[i], marker="o", color =colors[i], label=(i+1))
    	total_nodes += np.array(nodes_k[i][:len(total_nodes)])
    avg_nodes = total_nodes/len(nodes_k)
    plt.plot(avg_nodes, marker="o", color ="black", label="Mean")
    plt.title(f"{dataset} Nodes")
    plt.legend()
    plt.savefig(f"{path}/{dataset}/nodes.png", dpi=150)
    plt.close()

