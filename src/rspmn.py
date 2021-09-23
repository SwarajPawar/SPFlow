import numpy as np
from spn.algorithms.SPMN import SPMN
from spn.algorithms.EM import EM_optimization
import metaData, readData
from spn.structure.Base import Sum, Product, Max
from spn.structure.leaves.spmnLeaves.SPMNLeaf import State
from spn.structure.Base import assign_ids, rebuild_scopes_bottom_up, get_nodes_by_type
from spn.structure.StatisticalTypes import MetaType
from spn.algorithms.splitting.RDC import get_split_cols_RDC_py
from spn.algorithms.SPMNHelper import get_ds_context
import pandas as pd
from copy import deepcopy
from spn.algorithms.Inference import  likelihood
from spn.algorithms.MPE import mpe
from sklearn.feature_selection import chi2

dataset = "repeated_marbles"
debug = False
debug1 = True
plot = False
apply_em = False
use_chi2 = True
chi2_threshold = 0.005
likelihood_similarity_threshold = 0.00001
likelihood_match = True
deep_match = True
horizon = 3

problem_depth = 10

assert horizon <= problem_depth, "horizon cannot be greater than the problem_depth"

if dataset == "repeated_marbles":
    df = pd.DataFrame.from_csv("data/"+dataset+"/repeated_marbles_10000x20.tsv", sep='\t')
    data = df.values.reshape(10000,20,3)#[:5000,:6,:]
    nans=np.empty((data.shape[0],data.shape[1],1))
    nans[:] = np.nan
    train_data = np.concatenate((nans,data),axis=2)
    train_data[:,0,0]=0
    partialOrder = [['s1'],['draw'],['result','reward']]
    decNode=['draw']
    utilNode=['reward']
    scopeVars=['s1','draw','result','reward']
    scope = [i for i in range(len(scopeVars))]
    meta_types = [MetaType.STATE]+[MetaType.DISCRETE]*2+[MetaType.UTILITY]
elif dataset == "tiger":
    df = pd.DataFrame.from_csv("data/"+dataset+"/reverse_tiger_100000x10.tsv", sep='\t')
    data = df.values.reshape(100000,10,3)[:100000,:problem_depth,:]
    nans=np.empty((data.shape[0],data.shape[1],1))
    nans[:] = np.nan
    train_data = np.concatenate((nans,data),axis=2)
    train_data[:,0,0]=0
    partialOrder = [['s1'],['observation'],['action'],['reward']]
    decNode=['action']
    utilNode=['reward']
    scopeVars=['s1','observation','action','reward']
    scope = [i for i in range(len(scopeVars))]
    meta_types = [MetaType.STATE]+[MetaType.DISCRETE]*2+[MetaType.UTILITY]
elif dataset == "frozen_lake":
    df = pd.DataFrame.from_csv("data/"+dataset+"/frozen_lake_100000x10.tsv", sep='\t')
    data = df.values.reshape(100000,10,3)
    nans=np.empty((data.shape[0],data.shape[1],1))
    nans[:] = np.nan
    train_data = np.concatenate((nans,data),axis=2)
    train_data[:,0,0]=0
    partialOrder = [['s1'],['action'],['observation','reward']]
    decNode=['action']
    utilNode=['reward']
    scopeVars=['s1','action','observation','reward']
    scope = [i for i in range(len(scopeVars))]
    meta_types = [MetaType.STATE]+[MetaType.DISCRETE]*2+[MetaType.UTILITY]
elif dataset == "nchain":
    df = pd.DataFrame.from_csv("data/"+dataset+"/nchain_100000x10.tsv", sep='\t')
    data = df.values.reshape(100000,10,3)
    nans=np.empty((data.shape[0],data.shape[1],1))
    nans[:] = np.nan
    train_data = np.concatenate((nans,data),axis=2)
    train_data[:,0,0]=0
    partialOrder = [['s1'],['observation'],['action'],['reward']]
    decNode=['action']
    utilNode=['reward']
    scopeVars=['s1','observation','action','reward']
    scope = [i for i in range(len(scopeVars))]
    meta_types = [MetaType.STATE]+[MetaType.DISCRETE]*2+[MetaType.UTILITY]
elif dataset == "elevators":
    df = pd.read_csv("data/"+dataset+"/elevators_100000x10.tsv", index_col=0, sep='\t', header=None)
    data = df.values.reshape(100000,10,11)
    nans=np.empty((data.shape[0],data.shape[1],1))
    nans[:] = np.nan
    train_data = np.concatenate((nans,data),axis=2)
    train_data[:,0,0]=0
    decNode=[
            'close-door',
            'move-current-dir',
            'open-door-going-up',
            'open-door-going-down',
        ]
    obs = [
            'elevator-at-floor-0',
            'elevator-at-floor-1',
            'elevator-at-floor-2',
            'person-in-elevator-going-down',
            'elevator-dir',
            'person-waiting-1',
            'person-waiting-2',
            'person-waiting-3',
            'person-in-elevator-going-up',
        ]
    utilNode=['reward']
    #scopeVars=['s1']+obs+decNode+['reward']
    #partialOrder = [['s1'],obs]+[[x] for x in decNode]+[['reward']]
    scopeVars=['s1']+obs+decNode+['reward']
    partialOrder = [['s1']]+[obs]+[[x] for x in decNode]+[['reward']]
    scope = [i for i in range(len(scopeVars))]
    meta_types = [MetaType.STATE]+[MetaType.DISCRETE]*(len(obs)+len(decNode))+[MetaType.UTILITY]
    # in data, decisions come before obs. Move decisions after obs.
    #permutation = [0,5,6,7,8,9,1,2,3,4,10]
    #train_data[:] = train_data[:,permutation]
elif dataset == "elevators_mdp":
    df = pd.DataFrame.from_csv("data/"+dataset+"/elevators_mdp_100000x10.tsv", sep='\t', header=None)
    data = df.values.reshape(100000,10,18)
    nans=np.empty((data.shape[0],data.shape[1],1))
    nans[:] = np.nan
    train_data = np.concatenate((nans,data),axis=2)
    train_data[:,0,0]=0
    decNode=[
            'close-door',
            'move-current-dir',
            'open-door-going-up',
            'open-door-going-down',
        ]
    obs = [
            'elevator-closed[$e0]',
            'person-in-elevator-going-down[$e0]',
            'elevator-at-floor[$e0, $f0]',
            'elevator-at-floor[$e0, $f1]',
            'elevator-at-floor[$e0, $f2]',
            'person-waiting-up[$f0]',
            'person-waiting-up[$f1]',
            'person-waiting-up[$f2]',
            'elevator-dir-up[$e0]',
            'person-waiting-down[$f0]',
            'person-waiting-down[$f1]',
            'person-waiting-down[$f2]',
            'person-in-elevator-going-up[$e0]',
        ]
    utilNode=['reward']
    scopeVars=['s1']+obs+decNode+['reward']
    partialOrder = [['s1']]+[obs]+[[x] for x in decNode]+[['reward']]
    scope = [i for i in range(len(scopeVars))]
    meta_types = [MetaType.STATE]+[MetaType.DISCRETE]*(len(obs)+len(decNode))+[MetaType.UTILITY]
elif dataset == "skill_teaching":
    df = pd.read_csv("data/"+dataset+"/skill_teaching_100000x10.tsv", index_col=0, sep='\t', header=None)
    data = df.values.reshape(100000,10,13)
    nans=np.empty((data.shape[0],data.shape[1],1))
    nans[:] = np.nan
    train_data = np.concatenate((nans,data),axis=2)
    train_data[:,0,0]=0
    decNode=[
            'giveHint-1',
            'giveHint-2',
            'askProb-1',
            'askProb-2',
        ]
    obs = [
            'hintedRightObs-1',
            'hintedRightObs-2',
            'answeredRightObs-1',
            'answeredRightObs-2',
            'updateTurnObs-1',
            'updateTurnObs-2',
            'hintDelayObs-1',
            'hintDelayObs-2',
        ]
    utilNode=['reward']
    #scopeVars=['s1']+obs+decNode+['reward']
    #partialOrder = [['s1'],obs]+[[x] for x in decNode]+[['reward']]
    scopeVars=['s1']+obs+decNode+['reward']
    partialOrder = [['s1']]+[obs]+[[x] for x in decNode]+[['reward']]
    scope = [i for i in range(len(scopeVars))]
    meta_types = [MetaType.STATE]+[MetaType.DISCRETE]*(len(obs)+len(decNode))+[MetaType.UTILITY]
    # in data, decisions come before obs. Move decisions after obs.
    #permutation = [0,5,6,7,8,1,2,3,4,9]
    #train_data[:] = train_data[:,permutation]
elif dataset == "skill_teaching_mdp":
    df = pd.DataFrame.from_csv("data/"+dataset+"/skill_teaching_mdp_100000x10.tsv", sep='\t', header=None)
    data = df.values.reshape(100000,10,17)
    nans=np.empty((data.shape[0],data.shape[1],1))
    nans[:] = np.nan
    train_data = np.concatenate((nans,data),axis=2)
    train_data[:,0,0]=0
    decNode=[
            'giveHint-1',
            'giveHint-2',
            'askProb-1',
            'askProb-2',
        ]
    obs = [
            'hintDelayVar[$s0]',
            'hintDelayVar[$s1]',
            'updateTurn[$s0]',
            'updateTurn[$s1]',
            'answeredRight[$s0]',
            'answeredRight[$s1]',
            'proficiencyMed[$s0]',
            'proficiencyMed[$s1]',
            'proficiencyHigh[$s0]',
            'proficiencyHigh[$s1]',
            'hintedRight[$s0]',
            'hintedRight[$s1]',
        ]
    utilNode=['reward']
    scopeVars=['s1']+obs+decNode+['reward']
    partialOrder = [['s1']]+[obs]+[[x] for x in decNode]+[['reward']]
    scope = [i for i in range(len(scopeVars))]
    meta_types = [MetaType.STATE]+[MetaType.DISCRETE]*(len(obs)+len(decNode))+[MetaType.UTILITY]
elif dataset == "crossing_traffic":
    df = pd.read_csv("data/"+dataset+"/crossing_traffic_100000x10.tsv", sep='\t', index_col=0, header=None)
    data = df.values.reshape(100000,10,17)
    nans=np.empty((data.shape[0],data.shape[1],1))
    nans[:] = np.nan
    train_data = np.concatenate((nans,data),axis=2)
    train_data[:,0,0]=0
    decNode=[
            'move-east',
            'move-north',
            'move-south',
            'move-west'
        ]
    obs = [
            'arrival-max-xpos-1',
            'arrival-max-xpos-2',
            'arrival-max-xpos-3',
            'robot-at[$x1, $y1]',
            'robot-at[$x1, $y2]',
            'robot-at[$x1, $y3]',
            'robot-at[$x2, $y1]',
            'robot-at[$x2, $y2]',
            'robot-at[$x2, $y3]',
            'robot-at[$x3, $y1]',
            'robot-at[$x3, $y2]',
            'robot-at[$x3, $y3]',
        ]
    utilNode=['reward']
    #scopeVars=['s1']+obs+decNode+['reward']
    #partialOrder = [['s1'],obs]+[[x] for x in decNode]+[['reward']]
    scopeVars=['s1']+obs+decNode+['reward']
    partialOrder = [['s1']]+[obs]+[[x] for x in decNode]+[['reward']]
    scope = [i for i in range(len(scopeVars))]
    meta_types = [MetaType.STATE]+[MetaType.DISCRETE]*(len(obs)+len(decNode))+[MetaType.UTILITY]
    # in data, decisions come before obs. Move decisions after obs.
    #permutation = [0,5,6,7,1,2,3,4,8]
    #train_data[:] = train_data[:,permutation]
elif dataset == "crossing_traffic_mdp":
    df = pd.DataFrame.from_csv("data/"+dataset+"/crossing_traffic_mdp_10000x10.tsv", sep='\t', header=None)
    data = df.values.reshape(10000,10,23)
    nans=np.empty((data.shape[0],data.shape[1],1))
    nans[:] = np.nan
    train_data = np.concatenate((nans,data),axis=2)
    train_data[:,0,0]=0
    decNode=[
            'move-east',
            'move-north',
            'move-south',
            'move-west'
        ]
    obs = [
            'robot-at[$x1, $y1]',
            'robot-at[$x1, $y2]',
            'robot-at[$x1, $y3]',
            'robot-at[$x2, $y1]',
            'robot-at[$x2, $y2]',
            'robot-at[$x2, $y3]',
            'robot-at[$x3, $y1]',
            'robot-at[$x3, $y2]',
            'robot-at[$x3, $y3]',
            'obstacle-at[$x1, $y1]',
            'obstacle-at[$x1, $y2]',
            'obstacle-at[$x1, $y3]',
            'obstacle-at[$x2, $y1]',
            'obstacle-at[$x2, $y2]',
            'obstacle-at[$x2, $y3]',
            'obstacle-at[$x3, $y1]',
            'obstacle-at[$x3, $y2]',
            'obstacle-at[$x3, $y3]',
        ]
    utilNode=['reward']
    scopeVars=['s1']+obs+decNode+['reward']
    partialOrder = [['s1']]+[obs]+[[x] for x in decNode]+[['reward']]
    scope = [i for i in range(len(scopeVars))]
    meta_types = [MetaType.STATE]+[MetaType.DISCRETE]*(len(obs)+len(decNode))+[MetaType.UTILITY]
elif dataset == "game_of_life_mdp":
    df = pd.DataFrame.from_csv("data/"+dataset+"/game_of_life_mdp_100000x10.tsv", sep='\t', header=None)
    data = df.values.reshape(100000,10,19)
    nans=np.empty((data.shape[0],data.shape[1],1))
    nans[:] = np.nan
    train_data = np.concatenate((nans,data),axis=2)
    train_data[:,0,0]=0
    decNode=[
            'set[$x1, $y1]',
            'set[$x1, $y2]',
            'set[$x1, $y3]',
            'set[$x2, $y1]',
            'set[$x2, $y2]',
            'set[$x2, $y3]',
            'set[$x3, $y1]',
            'set[$x3, $y2]',
            'set[$x3, $y3]',
        ]
    obs = [
            'alive[$x1, $y1]',
            'alive[$x1, $y2]',
            'alive[$x1, $y3]',
            'alive[$x2, $y1]',
            'alive[$x2, $y2]',
            'alive[$x2, $y3]',
            'alive[$x3, $y1]',
            'alive[$x3, $y2]',
            'alive[$x3, $y3]',
        ]
    utilNode=['reward']
    scopeVars=['s1']+obs+decNode+['reward']
    partialOrder = [['s1']]+[obs]+[[x] for x in decNode]+[['reward']]
    scope = [i for i in range(len(scopeVars))]
    meta_types = [MetaType.STATE]+[MetaType.DISCRETE]*(len(obs)+len(decNode))+[MetaType.UTILITY]


dec_indices = [i for i in range(len(scopeVars)) if scopeVars[i] in decNode]

def get_horizon_train_data(data, horizon):
    # following line should concat each timestep with the next 'horizon' timesteps
    train_data_h = np.concatenate([data[:,i:data.shape[1]-horizon+i+1] for i in range(horizon)],axis=2)
    # add nans for s1
    nans_h=np.empty((train_data_h.shape[0],train_data_h.shape[1],1))
    nans_h[:] = np.nan
    train_data_h = np.concatenate((nans_h,train_data_h),axis=2)
    return train_data_h

def get_horizon_train_data_new(data, horizon):
    # following line should concat each timestep with the next 'horizon' timesteps
    nans_h=np.empty(data.shape)
    nans_h[:,:,:] = np.nan
    data = np.concatenate((data,nans_h),axis=1)
    train_data_h = np.concatenate([data[:,i:10+i] for i in range(horizon)],axis=2)
    # add nans for s1
    nans_h=np.empty((train_data_h.shape[0],train_data_h.shape[1],1))
    nans_h[:] = np.nan
    train_data_h = np.concatenate((nans_h,train_data_h),axis=2)
    return train_data_h

# merge sequence steps based on horizon
train_data_h = get_horizon_train_data(data, horizon)
# s1 for step 1 is 0
train_data_h[:,0,0]=0

def get_horizon_params(partialOrder, decNode, utilNode, scopeVars, meta_types, horizon):
    partialOrder_h = [] + partialOrder
    for i in range(1,horizon):
        partialOrder_h += [[var+"_t+"+str(i) for var in s] for s in partialOrder[1:]]
    decNode_h = decNode+[decNode[j]+"_t+"+str(i) for i in range (1,horizon) for j in range(len(decNode))]
    utilNode_h = utilNode+[utilNode[j]+"_t+"+str(i) for i in range (1,horizon) for j in range(len(utilNode))]
    scopeVars_h = scopeVars + [var+"_t+"+str(i) for i in range (1,horizon) for var in scopeVars[1:]]
    meta_types_h = meta_types+meta_types[1:]*(horizon-1)
    return partialOrder_h, decNode_h, utilNode_h, scopeVars_h, meta_types_h

partialOrder_h, decNode_h, utilNode_h, scopeVars_h, meta_types_h = get_horizon_params(
        partialOrder, decNode, utilNode, scopeVars, meta_types, horizon
    )
dec_indices_h = [i for i in range(len(scopeVars_h)) if scopeVars_h[i] in decNode_h]

# start run timer
import time
start_time = time.perf_counter()

spmn0 = SPMN(
        partialOrder_h,
        decNode_h,
        utilNode_h,
        scopeVars_h,
        meta_types_h,
        cluster_by_curr_information_set=True,
        util_to_bin = False
    )
spmn0_structure = spmn0.learn_spmn(train_data_h[:,0], chi2_threshold)

spmn0_stoptime = time.perf_counter()
spmn0_runtime = spmn0_stoptime - start_time

print("\nspmn0 runtime:\t" + str(spmn0_runtime))
print("spmn0 nodes:\t" + str(len(get_nodes_by_type(spmn0_structure))))

from spn.algorithms.MEU import meu
input_data = np.array([0]+[np.nan]*30)
print(meu(spmn0_structure, [input_data]))


if plot:
    from spn.io.Graphics import plot_spn
    plot_spn(spmn0_structure, "plots/"+dataset+"/spmn0.png")


SID_to_s2 = dict()
s1_to_s2s = dict()

import queue
def replace_nextState_with_s2(spmn,s2_scope_idx,s2_count=1, SID_to_s2=SID_to_s2, s1_to_s2s=s1_to_s2s):
    s1 = spmn.children[0]
    s1_to_s2s[s1] = list()
    scope_t1 = {i for i in range(s2_scope_idx)}
    q = q = queue.Queue()
    q.put(spmn)
    while not q.empty():
        node = q.get()
        if isinstance(node, Product):
            terminal = False
            to_remove = []
            for child in node.children:
                # if the child has no variables from the first timestep
                if len(set(child.scope) & scope_t1) == 0:
                    # then remove it to be replaced with an s2 node
                    to_remove.append(child)
                    terminal = True
                else:
                    q.put(child)
            if terminal:
                for child in to_remove:
                    node.children.remove(child)
                new_s2 = State(
                        [s2_count,s2_count+1],
                        [1],
                        [s2_count],
                        scope=s2_scope_idx
                    )
                node.children.append(new_s2)
                SID_to_s2[s2_count] = new_s2
                s1_to_s2s[s1].append(new_s2)
                s2_count += 1
        elif isinstance(node, Max) or isinstance(node, Sum):
            for child in node.children:
                q.put(child)
    return spmn, s2_count

# TODO replace this by using a placeholder for s2 as last infoset in partial order,
#  --- then just replace that placeholder with method above
import queue
def assign_s2(spmn,s2_scope_idx,s2_count=1, SID_to_s2=SID_to_s2):
    s1 = spmn.children[0]
    q = queue.Queue()
    q.put(spmn)
    while not q.empty():
        node = q.get()
        if isinstance(node, Max) or isinstance(node, Sum):
            for child in node.children:
                if isinstance(node, Max) or isinstance(node, Sum) or isinstance(node, Product):
                    q.put(child)
                else:
                    node.children.remove(child)
                    new_s2 = State(
                            [s2_count,s2_count+1],
                            [1],
                            [s2_count],
                            scope=s2_scope_idx
                        )
                    SID_to_s2[s2_count] = new_s2
                    node.children.append(Product(
                            children=[
                                child,
                                new_s2
                            ]
                        ))
                    s2_count += 1
        elif isinstance(node, Product):
            is_terminal = True
            for child in node.children:
                if isinstance(child, Max) or isinstance(child, Sum):
                    is_terminal = False
            if is_terminal:
                new_s2 = State(
                        [s2_count,s2_count+1],
                        [1],
                        [s2_count],
                        scope=s2_scope_idx
                    )
                SID_to_s2[s2_count] = new_s2
                s1_to_s2s[s1].append(new_s2)
                node.children.append(new_s2)
                s2_count += 1
            else:
                for child in node.children:
                    q.put(child)
    return spmn, s2_count

def update_s_nodes(spmn,s2_scope_idx,s2_count):
    # TODO caching state nodes for this would speed things up
    nodes = get_nodes_by_type(spmn)
    for node in nodes:
        if type(node)==State:
            bin_repr_points = list(range(s2_count))
            breaks = list(range(s2_count+1))
            densities = []
            for i in range(s2_count):
                if i in node.bin_repr_points:
                    densities.append(node.densities[node.bin_repr_points.index(i)])
                else:
                    densities.append(0)
            node.bin_repr_points = bin_repr_points
            node.breaks = breaks
            node.densities = densities
    return spmn

# add unique state identifier nodes for terminal branches of the spmn
spmn0_structure, s2_count = replace_nextState_with_s2(spmn0_structure, len(scopeVars), s2_count=1) # s2 is last scope index
spmn0_structure = assign_ids(spmn0_structure)
spmn0_structure = rebuild_scopes_bottom_up(spmn0_structure)
# update state nodes to contain probabilities for all state values
spmn0_structure = update_s_nodes(spmn0_structure,len(scopeVars),s2_count)


if plot:
    from spn.io.Graphics import plot_spn
    plot_spn(spmn0_structure,"plots/"+dataset+"/spmn0_with_s2.png")















#################### < creating template network    ####################
# create template network by adding a sum node with state branches as children
spmn_t = SPMN(
        partialOrder,
        decNode,
        utilNode,
        scopeVars,
        meta_types,
        cluster_by_curr_information_set=True,
        util_to_bin = False
    )

spmn_t_structure = Sum(weights=[1],children=[spmn0_structure])
spmn_t_structure = assign_ids(spmn_t_structure)
spmn_t_structure = rebuild_scopes_bottom_up(spmn_t_structure)
####################    creating template network /> ####################

def set_new_s1_vals(train_data, spmn_t_structure, val_to_s_branch, SID_to_s2, t):
    nans=np.empty((train_data.shape[0],train_data.shape[1],1))
    nans[:] = np.nan
    # s1 at t is s2 at t-1
    train_data_s2 = np.concatenate((train_data,nans),axis=2)
    prev_step_data = train_data_s2[:,t-1]
    prev_s1s = np.unique(prev_step_data[:,0]).astype(int)
    for s1_val in prev_s1s:
        if s1_val == 0:
            state_structure = spmn_t_structure.children[0]
            state_structure = assign_ids(state_structure)
        else:
            linked_branches = val_to_s_branch[s1_val]
            counts = np.array(list(SID_to_s2[s1_val].interface_links.values()))
            child_weights = (counts/np.sum(counts)).tolist()
            state_structure = Sum(weights=child_weights,children=linked_branches)
            state_structure = assign_ids(state_structure)
            state_structure = rebuild_scopes_bottom_up(state_structure)
        new_s1s = mpe(state_structure, prev_step_data[prev_step_data[:,0]==s1_val])[:,len(scopeVars)]
        train_data[train_data[:,t-1,0]==s1_val,t,0] = new_s1s
    spmn_t_structure = assign_ids(spmn_t_structure)
    spmn_t_structure = rebuild_scopes_bottom_up(spmn_t_structure)
    #new_s1s = mpe(spmn_t_structure, prev_step_data)[:,len(scopeVars)]
    #train_data[:,t,0] = new_s1s
    return train_data

def get_branch_s1_vals(branch):
    s1_node_vals = np.array(branch.children[0].bin_repr_points)
    s1_node_nonzero = np.ceil(branch.children[0].densities).astype(bool)
    #branch_s1_vals is the set of states this branch represents
    return list(s1_node_vals[s1_node_nonzero].astype(int))



def matches_state_branch(new_val, branch, spmn_t_structure, train_data, train_data_h, mean_branch_likelihoods, use_chi2, chi2_threshold,
        likelihood_similarity_threshold, s1_to_s2s, val_to_s_branch, t, d):
    train_data_unrolled = train_data[:,:t+1].reshape((-1,train_data.shape[2]))
    likelihood_train_data = train_data_unrolled_t[train_data_unrolled_t[:,0]==new_val]
    if likelihood_train_data.shape[0] < 1: # TODO This shouldn't ever happen?
        assert False, "found 0 length new_val data slice for new_val == " + str(new_val)
    # adding s2s as nan to satisfy code
    nans_lh_data=np.empty((likelihood_train_data.shape[0],1))
    nans_lh_data[:] = np.nan
    likelihood_train_data = np.concatenate((likelihood_train_data,nans_lh_data),axis=1)
    # setting s1s to nan to avoid considering them in likelihood (as they will always be different)
    likelihood_train_data[:,0] = np.nan
    branch_s1_vals = get_branch_s1_vals(branch)
    if debug: print("\n\tstate, branch_index:\t"+str(new_val)+",\t"+str(spmn_t_structure.children.index(branch)))
    if debug: print("\tbranch_s1_vals:\t"+str(branch_s1_vals))
    testing_s1_vals = branch_s1_vals + [new_val]
    mask = np.isin(train_data_unrolled[:,0],testing_s1_vals)
    # select data corresponding to this node's states and the new state
    train_data_s1_selected = train_data_unrolled[mask]
    if use_chi2:
        # look for correlations between state and the other values
        # np.delete here to ignore s1 itself and all decision values
        min_chi2_pvalue = np.nanmin(chi2(
                np.abs(np.delete(train_data_s1_selected,[0]+dec_indices,axis=1)),
                train_data_s1_selected[:,0]
            )[1])
    if branch in mean_branch_likelihoods:
        mean_likelihood_branch = mean_branch_likelihoods[branch]
    else:
        likelihood_data_branch = train_data_unrolled[
                np.isin(train_data_unrolled[:,0],branch_s1_vals)
            ]
        nans_lh_data=np.empty((likelihood_data_branch.shape[0],1))
        nans_lh_data[:] = np.nan
        likelihood_data_branch = np.concatenate((likelihood_data_branch,nans_lh_data),axis=1)
        # setting s1s to nan to avoid considering them in likelihood (as they will always be different)
        likelihood_data_branch[:,0] = np.nan
        if debug: print("\t< start calculating likelihood for branch "+str(spmn_t_structure.children.index(branch)))
        mean_likelihood_branch = np.mean(likelihood(branch, likelihood_data_branch))
        if debug: print("\tend calculating likelihood for branch "+str(spmn_t_structure.children.index(branch))+ " >")
        mean_branch_likelihoods[branch] = mean_likelihood_branch
    likelihood_new = likelihood(branch, likelihood_train_data)
    mean_likelihood_new = np.mean(likelihood_new)
    min_likelihood_new = np.min(likelihood_new) if likelihood_train_data.shape[0] > 0 else np.nan
    mean_likelihood_similarity = min_likelihood_new/mean_likelihood_branch
    if debug: print("\tmean_likelihood similarity:\t" + str(mean_likelihood_similarity))
    if debug: print("\tmin_likelihood_new:\t"+str(min_likelihood_new))
    if use_chi2:
        if debug: print("\tmin_chi2_pvalue:\t" + str(min_chi2_pvalue))
    if (use_chi2 and min_chi2_pvalue < chi2_threshold)\
            or mean_likelihood_similarity < likelihood_similarity_threshold \
            or min_likelihood_new < (1/10**10):
        # if s1 is correlated with any other variables,
        # then the new value is a functionally different state
        return False, mean_likelihood_similarity, min_likelihood_new, min_chi2_pvalue
    elif d > 1 and deep_match:
        branch_s2s = s1_to_s2s[branch.children[0]]
        #data_copy = deepcopy(train_data)
        #set_new_s1_vals(data_copy, branch, val_to_s_branch, SID_to_s2, t)
        #for s2 in branch_s2s:
            #if len(s2.interface_links) > 0:
                # next_branch = list(s2.interface_links.keys())[0]
                # # TODO copy data and set new s1 values based on s2
                # if not matches_state_branch(new_val, next_branch, spmn_t_structure,
                #         train_data, train_data_h, mean_branch_likelihoods, # TODO need to update s1s for t+1 and pass data by value
                #         use_chi2, chi2_threshold, likelihood_similarity_threshold,
                #         s1_to_s2s, val_to_s_branch, t+1, d-1):
                #     return False
            #else:
        if debug: print("\n\t\tdeep matching step")
        train_data_h_unrolled = train_data_h[:,:t+1].reshape((-1,train_data_h.shape[2]))
        mask = np.isin(train_data_h_unrolled[:,0],testing_s1_vals)
        data_h_s1_selected = train_data_h_unrolled[mask]
        #print("testing_s1_vals:\t"+str(testing_s1_vals))
        #print("data_h_s1_selected:\n"+str(data_h_s1_selected))
        #print("new_val is in data_h_s1_selected:\t"+str(np.any(np.isin(data_h_s1_selected[:,0],new_val))))
        # np.delete here to ignore s1 itself and all decision values
        min_chi2_pvalue = np.nanmin(chi2(
                np.abs(np.delete(data_h_s1_selected,[0]+dec_indices_h,axis=1)),
                data_h_s1_selected[:,0]
            )[1])
        if debug: print("\t\tmin_chi2_pvalue:\t"+str(min_chi2_pvalue))
        if min_chi2_pvalue < chi2_threshold or np.isnan(min_chi2_pvalue):
            if debug: print("\t\tfailed deep match")
            return False, mean_likelihood_similarity, min_likelihood_new, min_chi2_pvalue
        if debug1:
            print("\t\tpassed deep match")
        if apply_em:
            if debug1: print("\t\t < start EM")
            nans_em = np.empty((train_data_s1_selected.shape[0],1))
            nans_em[:] = np.nan
            train_data_em = np.concatenate((train_data_s1_selected,nans_em),axis=1)
            EM_optimization(branch, train_data_em, iterations=1, skip_validation=True)
            if debug1: print("\t\tend EM />")
    return True, mean_likelihood_similarity, min_likelihood_new, min_chi2_pvalue





















# learn new sub spmn branches based on state values
s1_vals = {0}
val_to_s_branch = dict()
val_to_s_branch[0]=[spmn_t_structure.children[0]]
mean_branch_likelihoods = dict()
for t in range(1, train_data.shape[1]):
    if debug: print("\n\n\n\n\n\nt:\t"+str(t))
    if debug1: print("\tt = "+str(t)+", \t num branches = "+str(len(spmn_t_structure.children))+"\n")
    #################### < setting s1 values for next step     ####################
    train_data = set_new_s1_vals(train_data, spmn_t_structure, val_to_s_branch, SID_to_s2, t)
    ####################    setting s1 values for next step /> ####################
    new_s1s = train_data[:,t,0]
    train_data_unrolled = train_data[:,:t+1].reshape((-1,train_data.shape[2]))
    train_data_unrolled_old = train_data[:,:t].reshape((-1,train_data.shape[2]))
    train_data_unrolled_t = train_data[:,t].reshape((-1,train_data.shape[2]))
    # when horizon would go beyond last step, we reduce the horizon to learn the last few steps
    if t >= train_data_h.shape[1]:
        horizon -= 1
        train_data_h = get_horizon_train_data(data, horizon)
        partialOrder_h, decNode_h, utilNode_h, scopeVars_h, meta_types_h = get_horizon_params(
                partialOrder, decNode, utilNode, scopeVars, meta_types, horizon
            )
        dec_indices_h = [i for i in range(len(scopeVars_h)) if scopeVars_h[i] in decNode_h]
        train_data_h[:,:,0] = train_data[:,:train_data_h.shape[1],0] # keeping train_data_h s1s updated
    train_data_h[:,t,0] = train_data[:,t,0]
    train_data_h_unrolled = train_data_h[:,t].reshape((-1,train_data_h.shape[2]))
    new_s1_vals = set(np.unique(new_s1s).astype(int))#i for i in range(s2_count)} - s1_vals
    old_s1_vals = deepcopy(s1_vals)
    new_s1_vals = new_s1_vals.difference(old_s1_vals) # only use truly new s1s for matching
    s1_vals = s1_vals.union(new_s1_vals)
    for new_val in new_s1_vals:
        likelihood_train_data = train_data_unrolled_t[train_data_unrolled_t[:,0]==new_val]
        if likelihood_train_data.shape[0] < 1: # TODO This shouldn't ever happen?
            print("\n\nfound 0 length new_val data slice for new_val == " + str(new_val))
            continue
        # adding s2s as nan to satisfy code
        nans_lh_data=np.empty((likelihood_train_data.shape[0],1))
        nans_lh_data[:] = np.nan
        likelihood_train_data = np.concatenate((likelihood_train_data,nans_lh_data),axis=1)
        # setting s1s to nan to avoid considering them in likelihood (as they will always be different)
        likelihood_train_data[:,0] = np.nan
        if debug1: print("\n\n< start matching for "+str(new_val)+" ...")
        if debug1: print("\tt = "+str(t)+", \t num branches = "+str(len(spmn_t_structure.children))+"\n")
        # check if new s1 val is the same state as any existing states
        match_found = False
        linked_branches = list()
        if new_val in old_s1_vals:
            linked_branches = val_to_s_branch[new_val]
            counts = np.array(list(SID_to_s2[new_val].interface_links.values()))
            child_weights = (counts/np.sum(counts)).tolist()
            state_structure = Sum(weights=child_weights,children=linked_branches)
            state_structure = assign_ids(state_structure)
            state_structure = rebuild_scopes_bottom_up(state_structure)
            likelihood_new = likelihood(state_structure, likelihood_train_data)
            # likelihood_train_data_old = train_data_unrolled_old[train_data_unrolled_old[:,0]==new_val]
            # nans_old=np.empty((likelihood_train_data_old.shape[0],1))
            # nans_old[:]=np.nan
            # likelihood_train_data_old = np.concatenate((likelihood_train_data_old,nans_old),axis=1)
            # likelihood_old = likelihood(state_structure, likelihood_train_data_old)
            spmn_t_structure = assign_ids(spmn_t_structure)
            spmn_t_structure = rebuild_scopes_bottom_up(spmn_t_structure)
            min_likelihood_new = np.min(likelihood_new)
            # mean_likelihood_new = np.mean(likelihood_new)
            # likelihood_similarity = mean_likelihood_new/np.mean(likelihood_old)
            if min_likelihood_new > (1/10**10): #and likelihood_similarity > likelihood_similarity_threshold:
                # increase counts for each branch based on likelihoods
                branch_likelihoods = []
                for branch in linked_branches:
                    branch = assign_ids(branch)
                    branch_likelihoods.append(likelihood(branch, likelihood_train_data))
                mpe_branches = np.argmax(branch_likelihoods, axis=0)
                for i, branch in enumerate(linked_branches):
                    SID_to_s2[new_val].interface_links[branch.children[0]] += np.sum(mpe_branches==i)
                spmn_t_structure = assign_ids(spmn_t_structure)
                spmn_t_structure = rebuild_scopes_bottom_up(spmn_t_structure)
                continue
        match_found = False
        best_mean_similarity = 0
        best_min_new = 0
        best_min_chi2 = 0
        for child in spmn_t_structure.children:
            #################### < matching to existing states    ####################
            if child in linked_branches: continue # we've already checked these
            match_found, mean_likelihood_similarity, min_likelihood_new, min_chi2_pvalue = matches_state_branch(new_val, child, spmn_t_structure,
                    train_data, train_data_h, mean_branch_likelihoods,
                    use_chi2, chi2_threshold, likelihood_similarity_threshold,
                    s1_to_s2s, val_to_s_branch, t, horizon)
            best_mean_similarity = max(best_mean_similarity, mean_likelihood_similarity)
            best_min_new = max(best_min_new, min_likelihood_new)
            best_min_chi2 = max(best_min_chi2, min_chi2_pvalue)
            if match_found:
                #################### < adding new val to existing state    ####################
                child_s1_vals = get_branch_s1_vals(child)
                child_s1_node = child.children[0]
                densities = child_s1_node.densities
                count_child = 0 #np.sum(np.isin(train_data_unrolled[:,0],child_s1_vals))
                for s1_val in child_s1_vals:
                    if s1_val == 0:
                        count_s1 = train_data.shape[0] # 1 starting state for each sequence
                    else:
                        count_s1 = SID_to_s2[s1_val].interface_links[child.children[0]]
                    densities[s1_val] = count_s1
                    count_child += count_s1
                count_new = likelihood_train_data.shape[0]
                count_child += count_new
                for s1_val in child_s1_vals:
                    densities[s1_val] /= count_child
                densities[new_val] = count_new / count_child
                child_s1_node.densities = densities
                # link s2 for new_val to this s1 node, or update counts
                if child_s1_node in SID_to_s2[new_val].interface_links:
                    SID_to_s2[new_val].interface_links[child_s1_node] += likelihood_train_data.shape[0]
                else:
                    SID_to_s2[new_val].interface_links[child_s1_node] = likelihood_train_data.shape[0]
                if new_val in val_to_s_branch:
                    val_to_s_branch[new_val] += [child]
                else:
                    val_to_s_branch[new_val] = [child]
                # as each branch is created to model a different distribution,
                #   we can expect that no further matches will be found.
                ####################    adding new val to existing state /> ####################
                break
        if debug1: print("end matching for "+str(new_val)+" >")
        if not match_found: # if this new state represents a new distribution
            #################### < creating new branch for state    ####################
            # then create new child SPMN for the state
            if debug: print("\n\nnew state\t"+str(new_val)+"\n\n")
            if debug: print("best_mean_similarity:\t"+str(best_mean_similarity))
            if debug: print("best_min_new:\t"+str(best_min_new))
            if debug: print("best_min_chi2:\t"+str(best_min_chi2))
            new_spmn_data = train_data_h_unrolled[train_data_h_unrolled[:,0]==new_val]
            spmn_new_s1 = SPMN(
                    partialOrder_h,
                    decNode_h,
                    utilNode_h,
                    scopeVars_h,
                    meta_types_h,
                    cluster_by_curr_information_set=True,
                    util_to_bin = False
                )
            spmn_new_s1_structure = spmn_new_s1.learn_spmn(new_spmn_data, chi2_threshold)
            if horizon == 1:
                spmn_new_s1_structure, s2_count = assign_s2(spmn_new_s1_structure, len(scopeVars), s2_count=s2_count)
            else:
                spmn_new_s1_structure, s2_count = replace_nextState_with_s2(spmn_new_s1_structure, len(scopeVars), s2_count=s2_count)
            SID_to_s2[new_val].interface_links[spmn_new_s1_structure.children[0]] = new_spmn_data.shape[0]
            if new_val in val_to_s_branch:
                val_to_s_branch[new_val] += [spmn_new_s1_structure]
            else:
                val_to_s_branch[new_val] = [spmn_new_s1_structure]
            spmn_t_structure.children += [spmn_new_s1_structure]
            # update weights for each child SPMN
            weights = []
            for child in spmn_t_structure.children:
                child_s1_vals = get_branch_s1_vals(child)
                count_child = 0 #np.sum(np.isin(train_data_unrolled[:,0],child_s1_vals))
                for s1_val in child_s1_vals:
                    if s1_val == 0:
                        count_child += train_data.shape[0] # 1 starting state for each sequence
                    else:
                        count_child += SID_to_s2[s1_val].interface_links[child.children[0]]
                prob_child = count_child / train_data_unrolled.shape[0]
                weights.append(prob_child)
            normalized_weights = np.array(weights) / np.sum(weights)
            spmn_t_structure.weights = normalized_weights.tolist()
            spmn_t_structure = update_s_nodes(spmn_t_structure, len(scopeVars), s2_count)
            spmn_t_structure = assign_ids(spmn_t_structure)
            spmn_t_structure = rebuild_scopes_bottom_up(spmn_t_structure)
            ####################    creating new branch for state /> ####################

#tune weights with EM
nans_em = np.empty((train_data_unrolled.shape[0],1))
nans_em[:] = np.nan
train_data_em = np.concatenate((train_data_unrolled,nans_em),axis=1)
EM_optimization(spmn_t_structure, train_data_em, skip_validation=True)

# stop run timer
end_time = time.perf_counter()
runtime = end_time - start_time

print("\nruntime:\t" + str(runtime))
print("nodes:\t" + str(len(get_nodes_by_type(spmn_t_structure))))
print()

spmn_t.spmn_structure = spmn_t_structure
rspmn = deepcopy(spmn_t)

file = open("data/"+str(dataset)+'/rspmn_'+'531'+'.pkle','wb')
import pickle
pickle.dump(rspmn, file)
file.close()

if plot:
    from spn.io.Graphics import plot_spn
    plot_spn(spmn_t_structure,  "plots/"+dataset+"/spmn_t.png", draw_interfaces=False)


# def unroll_rspmn(rspmn_root, depth):
#     #identify branches based on interface links
#     root = deepcopy(rspmn_root)
#     nodes = get_nodes_by_type(root)
#     inteface_to_branch_dict = dict()
#     for node in nodes:
#         if type(node)==State and len(node.interface_links)==1 and\
#             node.interface_links[0].id not in inteface_to_branch_dict:
#             for child in root.children:
#                 # if the interface link leads to this child's s1 node
#                 if node.interface_links[0] == child.children[0]:
#                     # then this s2 node leads to this branch
#                     # -- the actual branch is the sibling of the s1 node
#                     inteface_to_branch_dict[node.interface_links[0].id] = deepcopy(child.children[1])
#                     break
#     recursively_replace_s2_with_branch(root, inteface_to_branch_dict, depth-1)
#     root = assign_ids(root)
#     return root
#
# def recursively_replace_s2_with_branch(root, inteface_to_branch_dict, remaining_depth):
#     if remaining_depth == 0:
#         return
#     queue = [root]
#     while len(queue) > 0:
#         node = queue.pop(0)
#         if type(node) is Product or type(node) is Sum or type(node) is Max:
#             for i in range(len(node.children)):
#                 child = node.children[i]
#                 if type(child) is State and len(child.interface_links)==1:
#                     node.children[i] = deepcopy(inteface_to_branch_dict[child.interface_links[0].id])
#                     root = assign_ids(root)
#                     root = rebuild_scopes_bottom_up(root)
#                     recursively_replace_s2_with_branch(
#                             node.children[i],
#                             inteface_to_branch_dict,
#                             remaining_depth-1
#                         )
#                 elif type(child) is Product or type(child) is Sum or type(child) is Max:
#                     queue.append(child)
#
#
# unroll_len = 2
# rspmn2 = unroll_rspmn(spmn_t_structure, unroll_len)
#
# if plot:
#     from spn.io.Graphics import plot_spn
#     plot_spn(rspmn2,  "plots/"+dataset+"/unroll"+str(unroll_len)+".png")
#
# rspmn = deepcopy(spmn_t)
#
# # test meu
from spn.algorithms.MEU import rmeu
input_data = np.array([0]+[np.nan]*30)
print(rmeu(rspmn, input_data, depth=problem_depth))
#
#
# # load flspmn (reset caches)
# file = open("data/"+str(dataset)+'/rspmn_'+'531'+'.pkle','rb')
# import pickle
# rspmn = pickle.load(file)
# file.close()
# rmeu(flrspmn, input_data, depth=6)
#
# # inspect utilities
# from spn.structure.leaves.spmnLeaves.SPMNLeaf import Utility
# nodes = get_nodes_by_type(rspmn.spmn_structure)
# util_vals = [node.bin_repr_points for node in nodes if isinstance(node,Utility)]
# state_nodes = [node for node in nodes if isinstance(node,State)]
# max_nodes = [node for node in nodes if isinstance(node,Max)]
#
######################## tune weights with EM ##################################
nans_em = np.empty((train_data_unrolled.shape[0],1))
nans_em[:] = np.nan
train_data_em = np.concatenate((train_data_unrolled,nans_em),axis=1)
for branch in rspmn.spmn_structure.children:
    branch_SIDs = np.nonzero(branch.children[0].densities)[0]
    branch_data = train_data_em[np.isin(train_data_em[:,0], branch_SIDs)]
    try:
        assign_ids(branch)
        rebuild_scopes_bottom_up(branch)
        EM_optimization(branch, branch_data, iterations=3, skip_validation=False)
    except MemoryError:
        print("oof")
        error = True
        partitions = 1
        while(error):
            partitions *= 10
            error = False
            start = 0
            for i in range(partitions):
                if i < partitions-1:
                    end = start + train_data_em.shape[0] // partitions
                else:
                    end = -1
                try:
                    assign_ids(branch)
                    EM_optimization(branch, train_data_em[start:end], iterations=3, skip_validation=True)
                    start = end
                except MemoryError:
                    error = True
                    break

assign_ids(rspmn.spmn_structure)
rebuild_scopes_bottom_up(rspmn.spmn_structure)
