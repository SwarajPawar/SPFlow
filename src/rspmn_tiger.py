#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from spn.algorithms.SPMN import SPMN
from spn.algorithms.EM import EM_optimization
from spn.structure.Base import Sum, Product, Max
from spn.structure.leaves.spmnLeaves.SPMNLeaf import State, Utility
from spn.structure.Base import assign_ids, rebuild_scopes_bottom_up, get_nodes_by_type, Context
from spn.structure.StatisticalTypes import MetaType
from spn.algorithms.splitting.RDC import get_split_cols_RDC_py
from spn.algorithms.SPMNHelper import get_ds_context
import pandas as pd
from copy import deepcopy
from spn.algorithms.Inference import  likelihood
from spn.algorithms.MPE import mpe
from sklearn.feature_selection import mutual_info_classif
import queue, time
from datetime import datetime
import argparse
from spn.io.Graphics import plot_spn
from spn.algorithms.MEU import meu
from sklearn.preprocessing import normalize

class S_RSPMN:
    def __init__(self,
                dataset = "crossing_traffic",
                debug = False,
                debug1 = True,
                apply_em = False,
                mi_threshold = 0.01,
                deep_match = True,
                horizon = 3,
                problem_depth = 10,
                samples = 100000,
                num_vars = None,
                plot_path = "data"
            ):
        self.dataset = dataset
        self.debug = debug
        self.debug1 = debug1
        self.apply_em = apply_em
        self.mi_threshold = mi_threshold
        self.deep_match = deep_match
        self.horizon = horizon
        self.problem_depth = problem_depth
        self.samples = samples
        self.num_vars = num_vars
        self.plot_path = plot_path

        self.s1_node_to_SIDs = dict()
        self.SID_to_branch = dict()
        self.branch_to_SIDs = dict()
        self.SID_to_s2 = dict()
        self.s1_to_s2s = dict()

        if dataset == "skill_teaching_rl":
            self.meta_types = [MetaType.STATE]+[MetaType.DISCRETE]*(num_vars-3)+[MetaType.REAL]+[MetaType.DISCRETE]+[MetaType.UTILITY]
        else:
            self.meta_types = [MetaType.STATE]+[MetaType.DISCRETE]*(num_vars-1)+[MetaType.UTILITY]
        self.scope = [i for i in range(len(self.meta_types))]
        self.s2_count = 0
        self.s2_scope_idx = len(self.scope)
        self.spmn = None

        if dataset == "repeated_marbles":
            partialOrder = [['s1'],['draw'],['result','reward']]
            decNode=['draw']
            utilNode=['reward']
            scopeVars=['s1','draw','result','reward']
        elif dataset == "tiger":
            partialOrder = [['s1'],['observation'],['action'],['reward']]
            decNode=['action']
            utilNode=['reward']
            scopeVars=['s1','observation','action','reward']
        elif dataset == "frozen_lake":
            partialOrder = [['s1'],['action'],['observation','reward']]
            decNode=['action']
            utilNode=['reward']
            scopeVars=['s1','action','observation','reward']
        elif dataset == "nchain":
            partialOrder = [['s1'],['action'],['observation'],['reward']]
            decNode=['action']
            utilNode=['reward']
            scopeVars=['s1','action','observation','reward']
        elif dataset == "elevators":
            decNode=['decision']
                #     'close-door',
                #     'move-current-dir',
                #     'open-door-going-up',
                #     'open-door-going-down',
                # ]
            obs = [
                    # 'elevator-at-floor-0',
                    # 'elevator-at-floor-1',
                    'elevator-floor',# 'elevator-at-floor-2',
                    'person-in-elevator-going-down',
                    'elevator-dir',
                    #'person-waiting-1',
                    'person-waiting',#person-waiting-2',
                    #'person-waiting-3',
                    'person-in-elevator-going-up',
                ]
            utilNode=['reward']
            scopeVars=['s1']+decNode+obs+['reward']
            partialOrder = [['s1']]+[[x] for x in decNode]+[obs+['reward']]
        elif dataset == "skill_teaching":
            decNode=['decision']
                #     'giveHint-1',
                #     'giveHint-2',
                #     'askProb-1',
                #     'askProb-2',
                # ]
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
            scopeVars=['s1']+decNode+obs+['reward']
            partialOrder = [['s1']]+[[x] for x in decNode]+[obs+['reward']]
        elif dataset == "skill_teaching_rl":
            decNode=['action']
                #     'giveHint-1',
                #     'giveHint-2',
                #     'askProb-1',
                #     'askProb-2',
                # ]
            obs = [
                    'action-1',
                    'hintedRightObs-1',
                    'hintedRightObs-2',
                    'answeredRightObs-1',
                    'answeredRightObs-2',
                    'updateTurnObs-1',
                    'updateTurnObs-2',
                    'hintDelayObs-1',
                    'hintDelayObs-2',
                    'reward-1'
                ]
            utilNode=['reward']
            #scopeVars=['s1']+obs+decNode+['reward']
            #partialOrder = [['s1'],obs]+[[x] for x in decNode]+[['reward']]
            scopeVars=['s1']+obs+decNode+utilNode
            partialOrder = [['s1'],obs,decNode,utilNode]
        elif dataset == "crossing_traffic":
            decNode=['decision']
                #     'move-east',
                #     'move-north',
                #     'move-south',
                #     'move-west'
                # ]
            # obs = [
            #         'arrival-max-xpos-1',
            #         'arrival-max-xpos-2',
            #         'arrival-max-xpos-3',
            #         'robot-at[$x1, $y1]',
            #         'robot-at[$x1, $y2]',
            #         'robot-at[$x1, $y3]',
            #         'robot-at[$x2, $y1]',
            #         'robot-at[$x2, $y2]',
            #         'robot-at[$x2, $y3]',
            #         'robot-at[$x3, $y1]',
            #         'robot-at[$x3, $y2]',
            #         'robot-at[$x3, $y3]',
            #     ]
            utilNode=['reward']
            #scopeVars=['s1']+decNode+obs+['reward']
            scopeVars = ['s1', 'robot_position', 'decision',  'arrival', 'reward']
            #partialOrder = [['s1']]+[[x] for x in decNode]+[obs]+[['reward']]
            partialOrder = [['s1'],['robot_position']]+[[x] for x in decNode]+[['arrival','reward']]
            scope = [i for i in range(len(scopeVars))]
        elif dataset == "crossing_traffic_rl":
            decNode=['action']
            utilNode=['reward']
            scopeVars = ['s1', 'action-1', 'robot_position-1', 'arrive-1', 'reward-1', 'action', 'reward']
            partialOrder = [['s1'],['action-1', 'robot_position-1', 'arrive-1', 'reward-1'],['action'],['reward']]
            scope = [i for i in range(len(scopeVars))]
        self.decNode = decNode
        #self.obs=obs
        self.utilNode = utilNode
        self.scopeVars = scopeVars
        self.partialOrder = partialOrder
        self.dec_indices = [i for i in range(len(scopeVars)) if scopeVars[i] in decNode]
        self.util_indices = [i for i in range(len(scopeVars)) if scopeVars[i] in utilNode]
        self.bug_flag = False

    def get_horizon_train_data(self, data, horizon):
        nans_h=np.empty(data.shape)
        nans_h[:,:,:] = np.nan
        data = np.concatenate((data,nans_h),axis=1)
        train_data_h = np.concatenate([data[:,i:self.problem_depth+i] for i in range(horizon)],axis=2)
        # add nans for s1
        nans=np.empty((train_data_h.shape[0],train_data_h.shape[1],1))
        nans[:] = np.nan
        train_data_h = np.concatenate((nans,train_data_h),axis=2)
        return train_data_h

    def get_horizon_params(self,partialOrder, decNode, utilNode, scopeVars, meta_types, horizon):
        partialOrder_h = [] + partialOrder
        # for i in range(1,horizon):
        #     partialOrder_h += [[var+"_t+"+str(i) for var in s] for s in partialOrder[1:]]
        # decNode_h = decNode+[decNode[j]+"_t+"+str(i) for i in range (1,horizon) for j in range(len(decNode))]
        # utilNode_h = utilNode+[utilNode[j]+"_t+"+str(i) for i in range (1,horizon) for j in range(len(utilNode))]
        for i in range(1,horizon):
            partialOrder_h += [[var+str(i) for var in s] for s in partialOrder[1:]]
        decNode_h = decNode+[decNode[j]+str(i) for i in range (1,horizon) for j in range(len(decNode))]
        utilNode_h = utilNode+[utilNode[j]+str(i) for i in range (1,horizon) for j in range(len(utilNode))]
        scopeVars_h = [var for infoset in partialOrder_h for var in infoset]
        meta_types_h = meta_types+meta_types[1:]*(horizon-1)
        return partialOrder_h, decNode_h, utilNode_h, scopeVars_h, meta_types_h

    def replace_nextState_with_s2(self, spmn):
        s1 = spmn.children[0]
        self.s1_to_s2s[s1] = list()
        scope_t1 = {i for i in range(self.s2_scope_idx)}
        q = queue.Queue()
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
                            [self.s2_count,self.s2_count+1],
                            [1],
                            [self.s2_count],
                            scope=self.s2_scope_idx
                        )
                    self.SID_to_s2[self.s2_count] = deepcopy(new_s2)
                    node.children.append(self.SID_to_s2[self.s2_count])
                    self.s1_to_s2s[s1].append(self.SID_to_s2[self.s2_count])
                    self.s2_count += 1
            elif isinstance(node, Max) or isinstance(node, Sum):
                for child in node.children:
                    if len(set(child.scope) & scope_t1) == 0:
                        # then remove it to be replaced with an s2 node
                        node.children.remove(child)
                        new_s2 = State(
                                [self.s2_count,self.s2_count+1],
                                [1],
                                [self.s2_count],
                                scope=self.s2_scope_idx
                            )
                        self.SID_to_s2[self.s2_count] = deepcopy(new_s2)
                        node.children.append(self.SID_to_s2[self.s2_count])
                        self.s1_to_s2s[s1].append(self.SID_to_s2[self.s2_count])
                        self.s2_count += 1
                    else:
                        q.put(child)
        return spmn

    # TODO replace this by using a placeholder for s2 as last infoset in partial order,
    #  --- then just replace that placeholder using method above
    def assign_s2(self, spmn):
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
                                [self.s2_count,self.s2_count+1],
                                [1],
                                [self.s2_count],
                                scope=self.s2_scope_idx
                            )
                        self.SID_to_s2[self.s2_count] = new_s2
                        node.children.append(Product(
                                children=[
                                    child,
                                    new_s2
                                ]
                            ))
                        self.s2_count += 1
            elif isinstance(node, Product):
                is_terminal = True
                for child in node.children:
                    if isinstance(child, Max) or isinstance(child, Sum):
                        is_terminal = False
                if is_terminal:
                    new_s2 = State(
                            [self.s2_count,self.s2_count+1],
                            [1],
                            [self.s2_count],
                            scope=self.s2_scope_idx
                        )
                    self.SID_to_s2[s2_count] = new_s2
                    self.s1_to_s2s[s1].append(new_s2)
                    node.children.append(new_s2)
                    self.s2_count += 1
                else:
                    for child in node.children:
                        q.put(child)
        return spmn

    def update_s_nodes(self):
        nodes = get_nodes_by_type(self.spmn.spmn_structure)
        for node in nodes:
            if type(node)==State:
                bin_repr_points = list(range(self.s2_count))
                breaks = list(range(self.s2_count+1))
                densities = []
                for i in range(self.s2_count):
                    if i in node.bin_repr_points:
                        densities.append(node.densities[node.bin_repr_points.index(i)])
                    else:
                        densities.append(0)
                node.bin_repr_points = bin_repr_points
                node.breaks = breaks
                node.densities = densities

    def set_new_s1_vals(self, train_data, last_step_with_SID_idx, can_get_next_SID):
        nans=np.empty((train_data.shape[0],train_data.shape[1],1))
        nans[:] = np.nan
        # s1 at t is s2 at t-1
        train_data_s2 = np.concatenate((train_data,nans),axis=2)
        prev_step_data = train_data_s2[
                np.arange(train_data.shape[0]),
                last_step_with_SID_idx
            ]
        prev_SIDs = np.unique(prev_step_data[:,0]).astype(int)
        relevant_branches = list()
        for SID in prev_SIDs:
            if SID in self.SID_to_branch:
                branch = self.SID_to_branch[SID]
                if not branch in relevant_branches:
                    relevant_branches.append(branch)
        for branch in relevant_branches:
            branch_data_indices = np.arange(train_data.shape[0])[
                    np.logical_and(
                            can_get_next_SID,
                            np.isin(prev_step_data[:,0], self.branch_to_SIDs[branch])
                        )
                ]
            branch = assign_ids(branch)
            x = train_data_s2[
                    branch_data_indices,
                    last_step_with_SID_idx[branch_data_indices]
                ]
            #print(f"x.shape: {x.shape}")
            #print(f"branch: {branch}")
            #print(self.branch_to_SIDs[branch])
            new_SIDs = mpe(
                    branch,
                    train_data_s2[
                            branch_data_indices,
                            last_step_with_SID_idx[branch_data_indices]
                        ]
                )[:,self.s2_scope_idx]
            if np.any(np.isnan(new_SIDs)):
                print("\nfound nan SID assignment\n\n")
                new_SIDs[np.isnan(new_SIDs)] = -1
            train_data[
                    branch_data_indices,
                    last_step_with_SID_idx[branch_data_indices]+1,
                    0
                ] = new_SIDs
        self.spmn.spmn_structure = assign_ids(self.spmn.spmn_structure)
        #new_s1s = mpe(spmn_t_structure, prev_step_data)[:,len(scopeVars)]
        #train_data[:,t,0] = new_s1s
        return train_data



    def matches_state_branch(self, branch, train_data, SID_indices,
            last_step_with_SID_idx):
        branch_SIDs = self.branch_to_SIDs[branch]
        branch_SIDs_in_data = np.isin(train_data[:,:,0].astype(int),branch_SIDs)
        branch_sequence_indices = np.any(branch_SIDs_in_data, axis=1)
        branch_step_indices = np.argmax(branch_SIDs_in_data, axis=1)
        split_cols = get_split_cols_RDC_py(threshold=self.mi_threshold)
        for i in range(0,self.horizon):
            # select only sequences with sufficient remaining depth
            branch_sequence_indices_i = np.logical_and(
                    branch_sequence_indices,
                    (branch_step_indices+i)<self.problem_depth
                )
            SID_indices_i = np.logical_and(
                    SID_indices,
                    (last_step_with_SID_idx+i)<self.problem_depth
                )
            branch_data = train_data[
                    branch_sequence_indices_i,
                    branch_step_indices[branch_sequence_indices_i]
                ]
            newSID_data = train_data[
                    SID_indices_i,
                    last_step_with_SID_idx[SID_indices_i]
                ]
            for j in range(1,i+1):
                branch_data_j = train_data[
                        branch_sequence_indices_i,
                        branch_step_indices[branch_sequence_indices_i]+j
                    ][:,1:]
                newSID_data_j = train_data[
                        SID_indices_i,
                        last_step_with_SID_idx[SID_indices_i]+j
                    ][:,1:]
                branch_data = np.concatenate((branch_data[:,0].reshape(-1,1), branch_data_j), axis=1)
                newSID_data = np.concatenate((newSID_data[:,0].reshape(-1,1), newSID_data_j), axis=1)
            corr_test_data = np.append(newSID_data,branch_data,axis=0)
            if corr_test_data.shape[0] == 0: continue
            metatypes = self.meta_types# + self.meta_types[1:]*i
            ds_context = Context(meta_types=metatypes)
            ds_context.add_domains(corr_test_data)
            scope = self.scope#[j for j in range(len(self.scope) + len(self.scope[1:])*i)]
            #print("scope:\t"+str(scope))
            print("corr_test_data.shape:\t"+str(corr_test_data.shape))
            rdc_slices = split_cols(corr_test_data, ds_context, scope)
            for correlated_var_set_cluster, correlated_var_set_scope, weight in rdc_slices:
                if (0 in correlated_var_set_scope) and (len(correlated_var_set_scope) > 1):
                    return False
            # TODO: check to see if SID (scope 0) is clustered with any other variables
            #   if SID is only clustered with decision value then ignore
        print("match found!")
        return True




























    ################################ learn #####################################
    def learn_s_rspmn(self, data, plot = False):
        print("self.mi_threshold:\t"+str(self.mi_threshold))
        nans=np.empty((data.shape[0],data.shape[1],1))
        nans[:] = np.nan
        train_data = np.concatenate((nans,data),axis=2)
        train_data[:,0,0]=0
        print("got train_data")
        # merge sequence steps based on horizon
        train_data_h = self.get_horizon_train_data(data, self.horizon)
        print("got train_data_h")
        # s1 for step 1 is 0
        train_data_h[:,0,0]=0

        partialOrder_h, decNode_h, utilNode_h, scopeVars_h, meta_types_h = self.get_horizon_params(
                self.partialOrder, self.decNode, self.utilNode, self.scopeVars, self.meta_types, self.horizon
            )

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
        if True:
            print("start learning spmn0")
            spmn0_structure = spmn0.learn_spmn(train_data_h[:,0])
            spmn0_stoptime = time.perf_counter()
            spmn0_runtime = spmn0_stoptime - start_time
            print("learining spmn0 runtime:\t" + str(spmn0_runtime))
            print("spmn0 nodes:\t" + str(len(get_nodes_by_type(spmn0_structure))))
            file = open(f"spmn_0.pkle",'wb')
            import pickle
            pickle.dump(spmn0_structure, file)
            file.close()
        else:
            file = open(f"spmn_0.pkle",'rb')
            import pickle
            spmn0_structure = pickle.load(file)
            file.close()

        print("spmn0 meu:\t"+str(meu(spmn0_structure, np.array([[np.nan]*len(scopeVars_h)]))))

        if plot:
            from spn.io.Graphics import plot_spn
            print("plotting spmn0")
            plot_spn(spmn0_structure, f"{self.plot_path}/spmn0.png", feature_labels=scopeVars_h)

        self.s2_count = 1
        spmn0_structure = self.replace_nextState_with_s2(spmn0_structure) # s2 is last scope index
        spmn0_structure = assign_ids(spmn0_structure)
        spmn0_structure = rebuild_scopes_bottom_up(spmn0_structure)
        # update state nodes to contain probabilities for all state values
        self.SID_to_branch[0] = spmn0_structure
        self.branch_to_SIDs[spmn0_structure] = [0]
        self.s1_node_to_SIDs[spmn0_structure.children[0]] = [0]

        if plot:
            from spn.io.Graphics import plot_spn
            print("plotting spmn0 with s2 nodes")
            plot_spn(spmn0_structure,f"{self.plot_path}/spmn0_with_s2.png", feature_labels=self.scopeVars+["s2"])

        spmn_t = SPMN(
                self.partialOrder,
                self.decNode,
                self.utilNode,
                self.scopeVars,
                self.meta_types,
                cluster_by_curr_information_set=True,
                util_to_bin = False
            )
        spmn_t_structure = Sum(weights=[1],children=[spmn0_structure])
        spmn_t_structure = assign_ids(spmn_t_structure)
        spmn_t_structure = rebuild_scopes_bottom_up(spmn_t_structure)
        spmn_t.spmn_structure = spmn_t_structure
        self.spmn = spmn_t
        self.update_s_nodes()

        done = False
        total_pushing_SIDs_time = 0
        total_time_learning_structures = 0
        total_time_matching = 0
        while True:
            current_total_runtime = time.perf_counter() - start_time
            print("\n\nruntime so far:\t" + str(current_total_runtime)+"\tnum_branches:\t"+str(len(self.spmn.spmn_structure.children)))
            percent_time_pushing_SIDs = (total_pushing_SIDs_time / current_total_runtime)*100
            print("percent_time_pushing_SIDs:\t%.2f" % percent_time_pushing_SIDs)
            percent_time_learning_structures = (total_time_learning_structures / current_total_runtime)*100
            print("percent_time_learning_structures:\t%.2f" % percent_time_learning_structures)
            percent_time_matching = (total_time_matching / current_total_runtime)*100
            print("percent_time_matching:\t%.2f" % percent_time_matching)
            # push sequences forward through the existing structure until they all
            #   reach an SID which has not yet been linked to a branch.
            start_pushing_SIDs_time = time.perf_counter()
            while True:
                last_step_with_SID_idx = (np.argmax(np.isnan(train_data[:,:,0]), axis=1)-1).astype(int)
                last_step_with_SID_idx[last_step_with_SID_idx==-1] = self.problem_depth-1
                remaining_steps = np.sum(np.isnan(train_data[:,:,0]),axis=1)
                last_step_already_modeled = np.isin(
                        train_data[
                                np.arange(train_data.shape[0]),
                                last_step_with_SID_idx
                            ][:,0],
                        list(self.SID_to_branch.keys())
                    )
                can_get_next_SID = np.logical_and(last_step_already_modeled, remaining_steps > 0)
                # if any sequences' last processed step have SIDs which match to
                #   existing branches and have steps remaining:
                if np.any(can_get_next_SID) and np.any(np.isnan(train_data[:,:,0])):
                    # get the next SID
                    train_data = self.set_new_s1_vals(train_data, last_step_with_SID_idx, can_get_next_SID)
                    train_data_h[:,:,0] = train_data[:,:,0]
                else:
                    break
            pushing_SIDs_time = time.perf_counter() - start_pushing_SIDs_time
            total_pushing_SIDs_time += pushing_SIDs_time
            # once all sequences have reached a stopping point, find the unlinked
            #   SID with the most data waiting behind it.
            max_data_val = 0
            max_val_SID = None
            max_val_SID_indices = None
            unmatched_SID = False
            for SID in range(1, self.s2_count):
                if not SID in self.SID_to_branch:
                    unmatched_SID = True
                    SID_indices = train_data[
                                np.arange(train_data.shape[0]),
                                last_step_with_SID_idx
                            ][:,0]==SID
                        #np.arange(train_data.shape[0])[
                        #     train_data[
                        #             np.arange(train_data.shape[0]),
                        #             last_step_with_SID_idx
                        #         ][:self.s2_scope_idx]==SID
                        # ]
                    SID_data_val = np.sum(remaining_steps[SID_indices]+1)
                    # print("SID_data_val:\t"+str(SID_data_val))
                    # if SID_data_val == 0:
                    #     print("0 val SID:\t"+str(SID))
                    #     print("self.SID_to_branch:\n"+str(self.SID_to_branch))
                    if SID_data_val > max_data_val:
                        max_data_val = SID_data_val
                        max_val_SID = SID
                        max_val_SID_indices = SID_indices
            if not np.any(last_step_with_SID_idx < self.problem_depth) or not unmatched_SID or max_data_val==0:
                break
            # look for an existing branch that adequately models the data corresponding
            #   to this SID
            matched = False
            print(f"\nstart matching for SID {max_val_SID}")
            print("max_data_val:\t"+str(max_data_val)+"\tremaining_data:\t"+str(np.sum(remaining_steps)))
            print("max_val_SID_indices:\t"+str(max_val_SID_indices))
            start_matching_time = time.perf_counter()
            for branch in self.spmn.spmn_structure.children:
                if self.matches_state_branch(branch, train_data, max_val_SID_indices,
                        last_step_with_SID_idx):
                    self.branch_to_SIDs[branch].append(max_val_SID)
                    branch_SIDs = self.branch_to_SIDs[branch]
                    branch_s1_node = branch.children[0]
                    densities = branch_s1_node.densities
                    for SID in branch_SIDs:
                        densities[SID] = 1
                    branch_s1_node.densities = densities
                    # link s2 for new_val to this s1 node
                    self.SID_to_s2[max_val_SID].interface_links[branch_s1_node] = np.sum(max_val_SID_indices)
                    self.s1_node_to_SIDs[branch_s1_node] = branch_SIDs
                    self.SID_to_branch[max_val_SID] = branch
                    weights = []
                    for child in self.spmn.spmn_structure.children:
                        child_s1_vals = self.branch_to_SIDs[child]
                        count_child = 0 #np.sum(np.isin(train_data_unrolled[:,0],child_s1_vals))
                        for s1_val in child_s1_vals:
                            if s1_val == 0:
                                count_child += train_data.shape[0] # 1 starting state for each sequence
                            else:
                                count_child += self.SID_to_s2[s1_val].interface_links[child.children[0]]
                        prob_child = count_child / (self.samples * self.problem_depth)
                        weights.append(prob_child)
                    normalized_weights = np.array(weights) / np.sum(weights)
                    self.spmn.spmn_structure.weights = normalized_weights.tolist()
                    matched = True
                    # as each branch is created to model a different distribution,
                    #   we can expect that no further matches will be found.
                    break
            matching_time = time.perf_counter() - start_matching_time
            total_time_matching += matching_time
            start_time_learning_structure = time.perf_counter()
            if not matched:
                ################ < creating new branch for state   #############
                h = self.horizon
                tdh = self.get_horizon_train_data(data, h)
                tdh[:,:,0] = train_data[:,:,0]
                while True:
                    new_spmn_data = tdh[max_val_SID_indices, last_step_with_SID_idx[max_val_SID_indices]]
                    new_spmn_sl_data = new_spmn_data[~np.any(np.isnan(new_spmn_data),axis=1)]
                    if new_spmn_sl_data.shape[0] > 100:
                        break
                    elif h <= 2:
                        print(f"\n\th=1 for {max_val_SID}\n")
                        h = 1
                        new_spmn_sl_data = train_data[max_val_SID_indices, last_step_with_SID_idx[max_val_SID_indices]]
                        print("new_spmn_sl_data.shape:\t"+str(new_spmn_sl_data.shape))
                        new_spmn_sl_data = np.concatenate((new_spmn_sl_data,np.ones((new_spmn_sl_data.shape[0],1))), axis=1)
                        print("new_spmn_sl_data.shape:\t"+str(new_spmn_sl_data.shape))
                        break
                    else:
                        h -= 1
                        tdh = self.get_horizon_train_data(data, h)
                        tdh[:,:,0] = train_data[:,:,0]
                new_spmn_em_data = train_data[max_val_SID_indices, last_step_with_SID_idx[max_val_SID_indices]]
                em_nans = np.empty((new_spmn_em_data.shape[0],1))
                new_spmn_em_data = np.concatenate((new_spmn_em_data,em_nans),axis=1)
                partialOrder_h, decNode_h, utilNode_h, scopeVars_h, meta_types_h = self.get_horizon_params(
                        self.partialOrder, self.decNode, self.utilNode, self.scopeVars, self.meta_types, h
                    )
                if h == 1:
                    partialOrder_h.append(["dummy"])
                    scopeVars_h.append("dummy")
                    meta_types_h.append(MetaType.DISCRETE)
                spmn_new_s1 = SPMN(
                        partialOrder_h,
                        decNode_h,
                        utilNode_h,
                        scopeVars_h,
                        meta_types_h,
                        cluster_by_curr_information_set=True,
                        util_to_bin = False
                    )
                branch_num = len(self.spmn.spmn_structure.children)
                percentage_of_data_sl = (new_spmn_sl_data.shape[0]/self.samples)*100
                percentage_of_data_em = (new_spmn_em_data.shape[0]/self.samples)*100
                print(f"\ncreating branch {branch_num} for SID {max_val_SID}, \npercentage of data for SL: {percentage_of_data_sl}%, \npercentage of data for EM: {percentage_of_data_em}%")
                remaining_data = np.sum(remaining_steps)
                print(f"total remaining data: {remaining_data}")
                # print("\nnew_spmn_data[:10]:\n"+str(new_spmn_data[:10]))
                # print("\nlast_step_with_SID_idx[:5]:\n"+str(last_step_with_SID_idx[:5]))
                # print("\ntrain_data[:5]:\n"+str(train_data[:5]))
                spmn_new_s1_structure = spmn_new_s1.learn_spmn(new_spmn_sl_data)
                if h > 1:
                    spmn_new_s1_structure = self.replace_nextState_with_s2(spmn_new_s1_structure)
                else:
                    print(f"\n\th = 1 for SID {max_val_SID}")
                    spmn_new_s1_structure = self.replace_nextState_with_s2(spmn_new_s1_structure)
                    # from spn.io.Graphics import plot_spn
                    # spmn_new_s1_structure = assign_ids(spmn_new_s1_structure)
                    # plot_spn(spmn_new_s1_structure, "replaced_dummies.png", feature_labels=self.scopeVars+["s2"])
                    # spmn_new_s1_structure = self.assign_s2(spmn_new_s1_structure)
                # print("perfoming EM optimization")
                # EM_optimization(spmn_new_s1_structure, new_spmn_em_data, iterations=1, skip_validation=True)
                branch_s1_node = spmn_new_s1_structure.children[0]
                self.SID_to_branch[max_val_SID] = spmn_new_s1_structure
                self.SID_to_s2[max_val_SID].interface_links[branch_s1_node] = np.sum(max_val_SID_indices)
                self.s1_node_to_SIDs[branch_s1_node] = [max_val_SID]
                self.branch_to_SIDs[spmn_new_s1_structure] = [max_val_SID]
                self.spmn.spmn_structure.children += [spmn_new_s1_structure]
                weights = []
                for child in self.spmn.spmn_structure.children:
                    child_s1_vals = self.branch_to_SIDs[child]
                    count_child = 0 #np.sum(np.isin(train_data_unrolled[:,0],child_s1_vals))
                    for s1_val in child_s1_vals:
                        if s1_val == 0:
                            count_child += train_data.shape[0] # 1 starting state for each sequence
                        else:
                            count_child += self.SID_to_s2[s1_val].interface_links[child.children[0]]
                    prob_child = count_child / (self.samples * self.problem_depth)
                    weights.append(prob_child)
                normalized_weights = np.array(weights) / np.sum(weights)
                self.spmn.spmn_structure.weights = normalized_weights.tolist()
                self.update_s_nodes()
                self.spmn.spmn_structure = assign_ids(self.spmn.spmn_structure)
                self.spmn.spmn_structure = rebuild_scopes_bottom_up(self.spmn.spmn_structure)
            time_learning_structure = time.perf_counter() - start_time_learning_structure
            total_time_learning_structures += time_learning_structure
        learn_s_rspmn_stoptime = time.perf_counter()
        learn_s_rspmn_runtime = learn_s_rspmn_stoptime - start_time
        print(f"\n\nlearn_s_rspmn runtime: {learn_s_rspmn_runtime}\n\n")
        self.learning_time = learn_s_rspmn_runtime
        num_nodes = len(get_nodes_by_type(self.spmn.spmn_structure))
        print(f"num nodes:\t {num_nodes}")
        if plot:
            from spn.io.Graphics import plot_spn
            plot_spn(self.spmn.spmn_structure, f"{self.plot_path}/s-rspmn.png", feature_labels=self.scopeVars+["s2"])
            plot_spn(self.spmn.spmn_structure, f"{self.plot_path}/s-rspmn_interfaces.png", feature_labels=self.scopeVars+["s2"], draw_interfaces=True)
        return train_data

























def get_branch_to_decisions_to_s2(rspmn_root):
    branch_and_decisions_to_s2 = dict()
    for branch in rspmn_root.children:
        queue = branch.children[1:]
        fill_branch_and_decisions_to_s2(branch_and_decisions_to_s2, queue, [branch])
    branch_to_decisions_to_s2s = dict()
    for branch_and_decisions, s2 in branch_and_decisions_to_s2.items():
        branch = branch_and_decisions[0]
        decision_path = branch_and_decisions[1:]
        if branch in branch_to_decisions_to_s2s:
            branch_to_decisions_to_s2s[branch][decision_path] = s2
        else:
            branch_to_decisions_to_s2s[branch] = {decision_path: s2}
    return branch_to_decisions_to_s2s

def fill_branch_and_decisions_to_s2(branch_and_decisions_to_s2, queue, path):
    while len(queue) > 0:
        node = queue.pop(0)
        if isinstance(node, Max):
            for i in range(len(node.dec_values)):
                dec_val_i = node.dec_values[i]
                child_i = node.children[i]
                fill_branch_and_decisions_to_s2(
                        branch_and_decisions_to_s2,
                        [child_i],
                        path+[dec_val_i]
                    )
        elif isinstance(node, State):
            if tuple(path) in branch_and_decisions_to_s2:
                branch_and_decisions_to_s2[tuple(path)] += [node]
            else:
                branch_and_decisions_to_s2[tuple(path)] = [node]
        elif isinstance(node, Product) or isinstance(node, Sum):
            for child in node.children:
                queue.append(child)









def rmeu(rspmn, input_data, depth, debug=False):
    assert not np.isnan(input_data[0]), "starting SID (input_data[0]) must be defined."
    root = rspmn.spmn.spmn_structure
    if not input_data[0] in rspmn.SID_to_branch:
        print("SID_to_branch cache miss")
        return 0
    branch = rspmn.SID_to_branch[input_data[0]]
    branch_s2s = rspmn.s1_to_s2s[branch.children[0]]
    work_branch = deepcopy(branch)
    if depth > 1 and len(branch_s2s[0].interface_links) == 0:
        print("\nOOF\n")
        return None # unlinked branches cannot be evaluated beyond depth 1
    work_branch = assign_ids(work_branch)
    # set up caches
    if not hasattr(rspmn,"branch_and_depth_to_rmeu"):
        branch_and_depth_to_rmeu = dict()
        setattr(rspmn,"branch_and_depth_to_rmeu",branch_and_depth_to_rmeu)
    max_EU = None
    # if unconditioned meu for this state branch and depth has already been cached, just return the cached value
    if np.all(np.isnan(input_data[1:])):
        if (branch, depth) in rspmn.branch_and_depth_to_rmeu:
            root = assign_ids(root)
            return rspmn.branch_and_depth_to_rmeu[(branch, depth)]
        elif depth == 1:
            max_EU = meu(work_branch, np.array([input_data])).reshape(-1)
            rspmn.branch_and_depth_to_rmeu[(branch, depth)] = max_EU
            root = assign_ids(root)
            return max_EU
    elif depth == 1:
        max_EU = meu(work_branch, np.array([input_data])).reshape(-1)
        root = assign_ids(root)
        return max_EU
    SID_to_util = dict()
    for s2 in branch_s2s:
        SID = np.argmax(s2.densities).astype(int)
        next_data = np.array([SID]+[np.nan]*(rspmn.num_vars+1))
        s2_value = rmeu(rspmn, next_data, depth-1)
        # b = rspmn.SID_to_branch[SID]
        # bnum = rspmn.spmn.spmn_structure.children.index(b)
        # print(f"Branch: {bnum},  depth: {depth-1},  s2_value: {s2_value}")
        if s2_value is None: SID_to_util[SID] = None
        else:
            SID_to_util[SID] = Utility(
                    [s2_value,s2_value+1],
                    [1],
                    [s2_value],
                    scope=rspmn.s2_scope_idx
                )
    q = work_branch.children[1:]
    while len(q) > 0:
        node = q.pop(0)
        if isinstance(node, Max) or isinstance(node, Sum) or isinstance(node, Product):
            for i in range(len(node.children)):
                if isinstance(node.children[i], State):
                    SID = np.argmax(node.children[i].densities).astype(int)
                    # print(f"SID: {SID},  depth: {depth}-1,  s2_value: {s2_value}")
                    node.children[i] = SID_to_util[SID]
                else:
                    q.append(node.children[i])
            for i in range(len(node.children)):
                if node.children[i] is None:
                    print("missing child")
                    _ = node.children.pop(i)
                    if isinstance(node, Sum):
                        _ = node.weights.pop(i)
                        node.weights = normalize(node.weights, norm="l1")
    work_branch = remove_unlinked(work_branch)
    work_branch = assign_ids(work_branch)
    max_EU = meu(work_branch, np.array([input_data]))
    if np.all(np.isnan(input_data[1:])):
        rspmn.branch_and_depth_to_rmeu[(branch, depth)] = max_EU
    return max_EU





def remove_unlinked(branch):
    q = branch.children[1:]
    while len(q) > 0:
        node = q.pop(0)
        if isinstance(node, Max) or isinstance(node, Product):
            for i in range(len(node.children)):
                q.append(node.children[i])
        elif isinstance(node, Sum):
            to_remove = []
            for i in range(len(node.children)):
                if terminal_sum_child(node.children[i]):
                    print("removing child")
                    to_remove = [i]+to_remove
                else:
                    q.append(node.children[i])
            for i in to_remove:
                _ = node.children.pop(i)
                _ = node.weights.pop(i)
    return branch

def terminal_sum_child(child):
    q = [child]
    while len(q) > 0:
        node = q.pop(0)
        if node is None: return True
        if isinstance(node, Max) or isinstance(node, Product):
            for i in range(len(node.children)):
                q.append(node.children[i])
        elif isinstance(node,Sum): return False



def clear_caches(rspmn):
    del rspmn.branch_and_depth_to_rmeu


def get_action(branch, SID, dec_indices, num_vars=17):
    for i in range(len(dec_indices)):
        input_data = np.array([[np.nan]*(num_vars-len(dec_indices))+[0]*len(dec_indices)+[np.nan,SID]])
        input_data[0][(dec_indices[i])] = 1
        if likelihood(branch, input_data) > 0.000001:
            return i
    return "noop"



def best_next_decision(rspmn, input_data, depth=1, in_place=False):
    root = rspmn.spmn.spmn_structure
    if in_place:
        data = input_data
    else:
        data = np.copy(input_data)
    nodes = get_nodes_by_type(root)
    dec_dict = {}
    # find all possible decision values
    for node in nodes:
        if type(node) == Max:
            if node.dec_idx in dec_dict:
                dec_dict[node.dec_idx].union(set(node.dec_values))
            else:
                dec_dict[node.dec_idx] = set(node.dec_values)
    next_dec_idx = None
    # find next undefined decision
    for idx in dec_dict.keys():
        if np.all(np.isnan(data[:,idx])):
            next_dec_idx = idx
            break
    assert next_dec_idx != None, "please assign all values of next decision to np.nan"
    # determine best decisions based on meu
    dec_vals = list(dec_dict[next_dec_idx])
    best_decisions = np.full((1,data.shape[0]),dec_vals[0])
    data[:,next_dec_idx] = best_decisions
    if depth == 1:
        meu_best = meu(root, data)
    else:
        meu_best = np.array([rmeu(rspmn, data[0], depth)])
    for i in range(1, len(dec_vals)):
        decisions_i = np.full((1,data.shape[0]), dec_vals[i])
        data[:,next_dec_idx] = decisions_i
        if depth == 1:
            meu_i = meu(root, data)
        else:
            meu_i = np.array([rmeu(rspmn, data[0], depth)])
        best_decisions = np.select([np.greater(meu_i, meu_best),True],[decisions_i, best_decisions])
        data[:,next_dec_idx] = best_decisions
        meu_best = np.maximum(meu_i,meu_best)
    return best_decisions



def hard_em(rspmn, train_data):
    train_data_unrolled = train_data.reshape((-1,train_data.shape[2]))
    nans_em = np.empty((train_data_unrolled.shape[0],1))
    nans_em[:] = np.nan
    train_data_em = np.concatenate((train_data_unrolled,nans_em),axis=1)
    print(f"{len(rspmn.spmn.spmn_structure.children)} children")
    for i in range(len(rspmn.spmn.spmn_structure.children)):
        print(f"child {i}")
        branch = rspmn.spmn.spmn_structure.children[i]
        _ = assign_ids(branch)
        branch_SIDs = rspmn.branch_to_SIDs[branch]
        branch_em_data = train_data_em[np.isin(train_data_em[:,0], branch_SIDs)]
        next_SIDs = mpe(branch, branch_em_data)[:,rspmn.s2_scope_idx]
        unique, counts = np.unique(next_SIDs, return_counts=True)
        next_SID_counts = dict(zip(unique, counts))
        q = branch.children[1:]
        update_weights_hard_em(q, next_SID_counts)


def update_weights_hard_em(q, next_SID_counts):
    sums = []
    count = 0
    while(len(q) > 0):
        node = q.pop(0)
        if isinstance(node, Max) or isinstance(node, Product):
            for i in range(len(node.children)):
                q.append(node.children[i])
        elif isinstance(node, Sum):
            sums += [node]
        elif isinstance(node, State):
            count += next_SID_counts[np.argmax(node.densities)]
    for sum_node in sums:
        sum_counts = []
        for i in range(len(sum_node.children)):
            sum_counts += [update_weights_hard_em([sum_node.children[i]], next_SID_counts)]
        sum_sum_counts = sum(sum_counts)
        for i in range(len(sum_node.children)):
            sum_node.weights[i] = sum_counts[i]/sum_sum_counts
        count += sum_sum_counts
    return count
















# In[3]:


#################################### main ######################################
class arg:
    def __init__(self, dataset, horizon, mi, depth, samples, num_vars):
        self.dataset = dataset
        self.debug=0
        self.plot=False
        self.apply_em=False
        self.mi_threshold=mi
        self.deep_match=True
        self.horizon=horizon
        self.problem_depth=depth
        self.samples=samples
        self.num_vars=num_vars





#################################### main ######################################

if __name__ == "__main__":
    
    args = arg('tiger', 2, 0.3, 3, 10000, 3)

    


# In[4]:


import os, sys
from os import path
plot_path = f"plots\\{args.dataset}\\{args.samples}x{args.problem_depth}"
print(plot_path)
if not path.exists(plot_path):
    try:
        os.makedirs(plot_path)
    except OSError:
        print ("Creation of the directory %s failed" % plot_path)
        sys.exit()

num_vars = args.num_vars
if args.dataset == "crossing_traffic":
    num_vars = 4
elif args.dataset == "elevators":
    num_vars = 7
elif args.dataset == "skill_teaching":
    num_vars = 10



# In[5]:


rspmn = S_RSPMN(
            dataset = args.dataset,
            debug = args.debug==2,
            debug1 = args.debug>0,
            apply_em = args.apply_em,
            mi_threshold = args.mi_threshold,
            deep_match = args.deep_match,
            horizon = args.horizon,
            problem_depth = args.problem_depth,
            samples = args.samples,
            num_vars = num_vars,
            plot_path = plot_path
        )


# In[6]:


if "rl" in args.dataset:
    datapath = f"data/{args.dataset}/{args.dataset}_{args.samples}x{args.problem_depth}x{1}.tsv"
else:
    datapath = f"data/{args.dataset}/{args.dataset}_{args.samples}x{args.problem_depth}.tsv"
df = pd.read_csv(
    datapath,
    index_col=0, sep='\t',
    header=0 if args.dataset=="repeated_marbles" or args.dataset=="tiger"  or args.dataset=="frozen_lake" or args.dataset=="nchain" or "rl" in args.dataset else None)
data = df.values.reshape(args.samples,args.problem_depth,args.num_vars)
data = np.around(data, decimals=2)

if args.dataset == "crossing_traffic":
    decisions = data[:,:,:4]
    decisions = np.concatenate((np.zeros((decisions.shape[0],decisions.shape[1],1)),decisions),axis=2)
    decisions = np.argmax(decisions,axis=2)
    robot_position = np.argmax(data[:,:,7:-1],axis=2)
    arrival = data[:,:,5].reshape(data.shape[0],-1,1)
    reward = data[:,:,-1]
    data = np.concatenate(
            (
                robot_position.reshape(data.shape[0],-1,1),
                decisions.reshape(data.shape[0],-1,1),
                arrival.reshape(data.shape[0],-1,1),
                reward.reshape(data.shape[0],-1,1),
            ),
            axis=2
        )
elif args.dataset == "elevators":
    decisions = data[:,:,:4]
    decisions = np.concatenate((np.zeros((decisions.shape[0],decisions.shape[1],1)),decisions),axis=2)
    decisions = np.argmax(decisions,axis=2)
    elevator_floor = np.argmax(data[:,:,4:7],axis=2)
    person_waiting = data[:,:,11]
    elevator_dir = data[:,:,7]
    person_in_elevator_down = data[:,:,8]
    person_in_elevator_up = data[:,:,-2]
    reward = data[:,:,-1]
    data = np.concatenate(
            (
                decisions.reshape(data.shape[0],-1,1),
                elevator_floor.reshape(data.shape[0],-1,1),
                person_in_elevator_down.reshape(data.shape[0],-1,1),
                elevator_dir.reshape(data.shape[0],-1,1),
                person_waiting.reshape(data.shape[0],-1,1),
                person_in_elevator_up.reshape(data.shape[0],-1,1),
                reward.reshape(data.shape[0],-1,1),
            ),
            axis=2
        )
elif args.dataset == "skill_teaching":
    decisions = data[:,:,:4]
    decisions = np.concatenate((np.zeros((decisions.shape[0],decisions.shape[1],1)),decisions),axis=2)
    decisions = np.argmax(decisions,axis=2)
    data = np.concatenate(
            (
                decisions.reshape(data.shape[0],-1,1),
                data[:,:,4:]
            ),
            axis=2
        )


# In[7]:


train_data = rspmn.learn_s_rspmn(data, plot = args.plot)



# In[8]:


date = str(datetime.date(datetime.now()))[-5:].replace('-','')
hour = str(datetime.time((datetime.now())))[:2]

import pickle
pkle_path = f"data/{args.dataset}/{args.samples}x{args.problem_depth}/t:{args.mi_threshold}_h:{args.horizon}"
if not path.exists(pkle_path):
    try:
        os.makedirs(pkle_path)
    except OSError:
        print ("Creation of the directory %s failed" % pkle_path)
        file = open(f"data/{args.dataset}/rspmn_{date}_{hour}.pkle",'wb')
        pickle.dump(rspmn, file)
        file.close()
        data_SIDs = train_data[:,:,0].reshape(args.samples,args.problem_depth)
        np.savetxt(f"data/{args.dataset}/data_SIDs_{date}_{hour}.tsv", data_SIDs, delimiter='\t')
if path.exists(pkle_path):
    file = open(f"{pkle_path}/rspmn_{date}_{hour}.pkle",'wb')
    pickle.dump(rspmn, file)
    file.close()
    data_SIDs = train_data[:,:,0].reshape(args.samples,args.problem_depth)
    np.savetxt(f"{pkle_path}/data_SIDs_{date}_{hour}.tsv", data_SIDs, delimiter='\t')



# In[9]:


input_data = np.array([0]+[np.nan]*(args.num_vars+1))
for i in range(1,args.problem_depth+1):
    print(f"rmeu for depth {i}:\t"+str(rmeu(rspmn, input_data, depth=i)))

print(f"rmeu for depth 100:\t"+str(rmeu(rspmn, input_data, depth=100)))



# In[10]:


print("\napplying EM\n")
clear_caches(rspmn)
rspmn = hard_em(rspmn, train_data)
# train_data_unrolled = train_data.reshape((-1,train_data.shape[2]))
# nans_em = np.empty((train_data_unrolled.shape[0],1))
# nans_em[:] = np.nan
# train_data_em = np.concatenate((train_data_unrolled,nans_em),axis=1)
# for i in range(len(rspmn.spmn.spmn_structure.children)):
#     _ = assign_ids(rspmn.spmn.spmn_structure.children[i])
#     branch_SIDs = rspmn.branch_to_SIDs[rspmn.spmn.spmn_structure.children[i]]
#     branch_em_data = train_data_em[np.isin(train_data_em[:,0], branch_SIDs)]
#     EM_optimization(rspmn.spmn.spmn_structure.children[i], branch_em_data, skip_validation=True, iterations=1)



# In[11]:


input_data = np.array([0]+[np.nan]*(args.num_vars+1))
for i in range(1,args.problem_depth+1):
    print(f"rmeu for depth {i}:\t"+str(rmeu(rspmn, input_data, depth=i)))

print(f"rmeu for depth 100:\t"+str(rmeu(rspmn, input_data, depth=100)))



# In[ ]:


if path.exists(pkle_path):
    file = open(f"{pkle_path}/rspmn_EM_{date}_{hour}.pkle",'wb')
    pickle.dump(rspmn, file)
    file.close()


# In[ ]:




