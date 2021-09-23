import gym
import numpy as np
from numpy.random import randint
from copy import deepcopy
import argparse
#from spn.algorithms.MEU import rmeu, best_next_decision
from spn.algorithms.SPMN import SPMN
from spn.algorithms.MPE import mpe
from spn.structure.Base import assign_ids
from rspmn_new import *

trials = 50000
steps = 100

slip = True
noisy = False

display = False

problem = "NChain"

if problem == "FrozenLake":
    env = gym.make("FrozenLake-v0",is_slippery=slip)
    file = open('data/frozen_lake/rspmn_407.pkle','rb')
elif problem == "NChain":
    env = gym.make("NChain-v0")
    file = open('data/nchain/100000x10/t:0.3_h:2/rspmn_0824_13.pkle','rb')
elif problem == "crossing_traffic":
    env = gym.make("CrossingTraffic-v0")
    file = open('data/crossing_traffic/rspmn_531.pkle','rb')
elif problem == "elevators":
    env = gym.make("Elevators-v0")
    file = open('data/crossing_traffic/rspmn_531.pkle','rb')
elif problem == "skill_teaching":
    env = gym.make("SkillTeaching-v0")

import gym
env = gym.make("CrossingTraffic-v0")
trials = 1000
avg_rwd = 0
# action_counts={0:0,1:0,2:0,3:0,4:0}
for i in range(trials):
    _ = env.reset()
    done = False
    trial_reward = 0
    t = 0
    road = False
    passed = False
    prev_obs = int(env._get_obs()[1])
    prev_prev_obs = prev_obs
    while not done:
        if t==0: action = 3
        elif not (road or passed):
            if prev_obs==1: action = 4
            else:
                action = 1
                road = True
        elif road and (not passed):
            action = 1
            passed = True
        else: action = 0
        #elif t==1 or t==2: action = 1
        #elif t==3: action = 0
        t+=1
        #action = env.action_space.sample()
        # action_counts[action]+=1
        prev_prev_obs = prev_obs
        prev_obs = env._get_obs()[1]
        state, reward, done, info = env.step(action)
        #print(f"state:\t{state}, reward:\t{reward}")
        trial_reward += reward
    #print(f"trial reward: {trial_reward}\n")
    avg_rwd += trial_reward

avg_rwd /= trials
print(f"avg_rwd: {avg_rwd}")
# east = action_counts[0]/trials; north = action_counts[1]/trials
# south = action_counts[2]/trials; west = action_counts[3]/trials
# noop = action_counts[4]/trials
# print(f"east:\t{east}\nnorth:\t{north}\nsouth:\t{south}\nwest:\t{west}\nnoop:\t{noop}\n")

import pickle
rspmn = pickle.load(file)
root = rspmn.spmn.spmn_structure

SID_to_branch = dict()
input_data_to_SID = dict()
SID_and_depth_to_action = dict()
input_and_depth_to_action = dict()
all_trials_reward = 0
avg_rwds = []
for i in range(trials):
    total_reward = 0
    _ = env.reset()
    if display:
        env.render()
    done = False
    t=0
    while not done and t<steps:
        if t == 0:
            SID = 0
            input_data = np.array([[SID]+[np.nan]*4])
        if input_data[0,0] in SID_and_depth_to_action:#input_and_depth_to_action:
            #action = input_and_depth_to_action[tuple(input_data[0])]
            action = SID_and_depth_to_action[input_data[0,0]]
        else:
            action = best_next_decision(rspmn, input_data, depth=steps-t).reshape(1).astype(int)[0]
            #input_and_depth_to_action[tuple(input_data[0])] = action
            SID_and_depth_to_action[input_data[0,0]] = action
        #action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        total_reward += reward
        input_data[0,1] = action
        input_data[0,2] = state
        input_data[0,3] = reward
        if not SID in rspmn.SID_to_branch:
            for branch in rspmn.spmn_structure.children:
                if branch.children[0].densities[SID] > 0:
                    rspmn.SID_to_branch[SID] = branch
        if tuple(input_data[0,:4]) not in input_data_to_SID:
            _ = assign_ids(rspmn.SID_to_branch[SID])
            input_data_to_SID[tuple(input_data[0,:4])] = mpe(rspmn.SID_to_branch[SID], input_data).astype(int)[0,-1]
            _ = assign_ids(root)
        SID = input_data_to_SID[tuple(input_data[0,:4])]
        input_data = np.array([[SID]+[np.nan]*4])
        # if display:
            # print("\t"+str([t,action,state,reward,SID]))
            # env.render()
        t+=1
    if display:
        print("\ntotal reward for trial " + str(i+1) + ":\t"+str(total_reward)+"\n\n")
    all_trials_reward += total_reward
    # if i>0 and (i+1)%100 == 0:
    #     print(f"{i+1} average_reward:\t" + str(all_trails_reward/i))
    if (i+1)%10000 == 0:
        avg_rwds+=[all_trials_reward/10000]
        all_trials_reward = 0
else:
num_actions = env.action_space.n
state_vars = len(env.observation_space.spaces)
all_trails_reward = 0
for i in range(trials):
    total_reward = 0
    _ = env.reset()
    if display:
        print("game: "+str(i))
        env.render()
    done = False
    t=0
    while not done and t<steps:
        if t == 0:
            SID = 0
        input_data = np.array([SID]+list(env._get_obs())+[np.nan]*(num_actions+2))
        best_decision_val = -100000000000000
        best_decision = None
        for j in range(num_actions):
            input_data[(state_vars+1):(state_vars+1+num_actions)] = 0
            input_data[(state_vars+j)] = 1
            # print(f"input_data: {input_data}")
            decision_val = rmeu(rspmn, input_data, depth=steps-t)
            #print(f'Action: {j}, MEU: {decision_val}')
            if decision_val > best_decision_val:
                best_decision_val = decision_val
                best_decision = j
        #print(f'Best action: {best_decision}, MEU: {best_decision_val}')
        input_data[(state_vars+1):(state_vars+1+num_actions)] = 0
        input_data[state_vars+1+best_decision] = 1
        state, reward, done, info = env.step(best_decision)
        input_data[state_vars+num_actions+1] = reward
        if not SID in SID_to_branch:
            for branch in rspmn.spmn_structure.children:
                if branch.children[0].densities[SID] > 0:
                    SID_to_branch[SID] = branch
        _ = assign_ids(SID_to_branch[SID])
        #mpe(SID_to_branch[SID], np.array([input_data])).astype(int)[0,state_vars+num_actions+2]
        _ = assign_ids(rspmn.spmn_structure)
        if tuple(input_data[:state_vars+num_actions+1]) not in input_data_to_SID:
            _ = assign_ids(SID_to_branch[SID])
            input_data_to_SID[tuple(input_data[:state_vars+num_actions+1])] = mpe(SID_to_branch[SID], np.array([input_data])).astype(int)[0,state_vars+num_actions+1]
            _ = assign_ids(rspmn.spmn_structure)
        SID = input_data_to_SID[tuple(input_data[:state_vars+num_actions+1])]
        total_reward += reward
        if display:
            #print("\t"+str([t,state,action,reward,SID]))
            env.render()
        t+=1
    #if display:
    print("\ntotal reward for trial " + str(i+1) + ":\t"+str(total_reward))
    all_trails_reward += total_reward
    if i>0 and i%100 == 0:
        print(f"{i} average_reward:\t" + str(all_trails_reward/i))


average_reward = all_trails_reward / trials
