

import numpy as np

import matplotlib.pyplot as plt

path = "cross_new"
datasets = ["nltcs", "msnbc", "plants", "kdd", "baudio", "jester", "bnetflix"]

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
    
    
original = {"nltcs":
	{"ll": [-6.4208788522751385], "n": [113]},
        "msnbc":
	{"ll": [-6.59705998562684],	"n": [42]},
        "plants":
	{"ll": [-13.87416406710313],		"n": [3819]},
        "kdd":
	{"ll": [-2.2653314792889714],		"n": [287]},
        "baudio":
	{"ll": [-43.359889361629534]	,	"n": [329]},
        "jester":
	{"ll": [-57.02821489294516]	,	"n": [255]},
        "bnetflix":
	{"ll": [-61.299623159930995]	,	"n": [338]}
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
    maxlen = max([len(lls[i]) for i in range(len(lls))])
    total_ll = np.zeros(min([len(lls[i]) for i in range(len(lls))]))
    upperll = [upper[dataset]["ll"]] * maxlen
    plt.plot(upperll, linestyle="dotted", color ="red", label="Upper Limit")
    originalll = [original[dataset]["ll"]] * maxlen
    plt.plot(originalll, linestyle="dotted", color ="blue", label="LearnSPN")
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
    uppern = [upper[dataset]["n"]] * maxlen
    plt.plot(uppern, linestyle="dotted", color ="red", label="Upper Limit")
    originaln = [original[dataset]["n"]] * maxlen
    plt.plot(originaln, linestyle="dotted", color ="blue", label="LearnSPN")
    for i in range(len(nodes_k)):
    	plt.plot(nodes_k[i], marker="o", color =colors[i], label=(i+1))
    	total_nodes += np.array(nodes_k[i][:len(total_nodes)])
    avg_nodes = total_nodes/len(nodes_k)
    plt.plot(avg_nodes, marker="o", color ="black", label="Mean")
    plt.title(f"{dataset} Nodes")
    plt.legend()
    plt.savefig(f"{path}/{dataset}/nodes.png", dpi=150)
    plt.close()

