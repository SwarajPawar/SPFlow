# RSPMN
Provides appendix, sample datasets used in experiments and instructions for using the implementation of RSPMN.
RSPMN module of SPFlow library implements the structure learning algorithm for Recurrent-Sum-Product-Max Networks(**RSPMN**)
which generalise Sum-Product-Max Networks(**SPN**) for the class of sequential decison-making problems. RSPMN can be used to claculate MEU and obtain policies on sequential decision making domains.

## Getting Started

Use the *rspmn* branch from forked version of SPFlow [https://github.com/SwarajPawar/SPFlow] for installation.

## Using RSPMN Module

#### Sample Dataset
Look at *RSPMN/RSPMN_MDP_Datasets/* folder for a list of sample datasets to use with RSPMN structure learning algorithm. 
*ReadMe.txt* of each data set contains meta data about datasets. It contains information about 
*partial order, decision nodes, utility node, meta_types, dataset size, optimal meu, etc* for each of the data sets.
```python
import pandas as pd    
csv_path = "FrozenLake/FrozenLake.csv"
df = pd.DataFrame.from_csv(csv_path, sep=',')
train_data = df.values
```
#### Provide initial parameters
Provide *partial order, decision nodes, utility node, feature_names, meta_types* for the dataset
```python
Partial Order = 
[[state],[action],[reward]]
decision_nodes = ["action"]
utility_nodes = ["reward"]
feature_names = [var for var_set in partial_order for var in var_set]

from spn.structure.StatisticalTypes import MetaType
# Utility variable is the last variable. Other variables are of discrete type
meta_types = [MetaType.DISCRETE]*2+[MetaType.UTILITY]
```
#### Learn the structure of RSPMN 

```python
from spn.algorithms.RSPMNnewAlgo import RSPMNnewAlgo
rspmn = RSPMNnewAlgo(partial_order, decision_nodes, utility_nodes, feature_names, 
                     meta_types, cluster_by_curr_information_set=True,
                     util_to_bin=False)
wrapped_two_timestep_data = rspmn.InitialTemplate.wrap_sequence_into_two_time_steps(train_data)
spmn_structure_two_time_steps, top_network, initial_template_network = rspmn.InitialTemplate.build_initial_template(wrapped_two_timestep_data)    
```
### Learn final template network from initial template network and top network
```python

template = rspmn.InitialTemplate.template_network

# use one of the follwoing ways to update the template

# Learn final template using sequential data
template = rspmn.hard_em(train_data, template, False)

# or 

# if numpy array cannot hold whole of train_data, 
# updates can be made on batches of data and/or by splitting sequence as follows
for i in range(0, len(train_data), batch_size):
    print(i)
    for j in range(0, len(sequence)-1, sequence_split): # e.g. len(sequence) = 10, sequence_split = 5
        print(j)
        template = rspmn.hard_em(train_data[i:i+batch_size, j*num_vars:(j+sequence_split)*num_vars], template, False)
        
rspmn.update_weights(rspmn.template)

```
#### We can plot the learned structures 
```python
from spn.io.Graphics import plot_spn
plot_spn(spmn_structure_two_time_steps, "folder/file_name.pdf", feature_labels=["State0", "Action0", "Reward0", "State1", "Action1", "Reward1"])
plot_spn(top_layer, "folder/file_name.pdf", feature_labels=["State", "Action", "Reward"])
plot_spn(initial_template_network, "folder/file_name.pdf", feature_labels=["State", "Action", "Reward"])

```

    
#### Calculating Maximum Expected Utility (MEU) and obtaining Best Decisions from the learned RSPMN
```python
meu_list, lls_list = rspmn.value_iteration(template, num_of_iterations)

# to select an action in state 0
test_data = [0, np.nan, np.nan]
test_data = np.array(test_data).reshape(1, len(test_data))
# rspmn.select_actions returns a numpy array of data filled with best decision values at corresponding decision variables
# and most probable explaination (MPE) values for leaf variables
data = rspmn.select_actions(rspmn.template, test_data, meu_list, lls_list)
# data is represented as [state, action, utility]
action = data[1]
# print(action)

# to obtain the Maximum Expected Utility of state 0
test_data = [0, np.nan, np.nan]
test_data = np.array(test_data).reshape(1, len(test_data))
meu = rspmn.meu_of_state(rspmn.template, test_data, meu_list, lls_list)[0][:,0]
# print(meu)
```    
The output for meu
```python
# after 300 iterations on value iteration
0.818
```
#### Additional functionality
We can also interact with the environment by using actions generated from RSPMN
```python
import gym
env = gym.make('FrozenLake-v0')
episode_rewards = [0.0]
obs = env.reset()
for i in range(num_steps):
  test_data = [obs, np.nan, np.nan]
  data = rspmn.select_actions(rspmn.template, test_data, meu_list, lls_list)
  # data is represented as [state, action, utility]
  action = data[1]
  obs, reward, done, info = env.step(action)
  
  # Stats
  episode_rewards[-1] += reward
  if done:
      obs = env.reset()
      episode_rewards.append(0.0)
# Compute mean reward for the last 100 episodes
mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
# print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))
```  
## Papers implemented
Tatavarti, Hari Teja, Prashant Doshi, and Layton Hayes. "Recurrent Sum-Product-Max Networks for Decision Making in Perfectly-Observed Environments." arXiv preprint arXiv:2006.07300 (2020).

