# SPMN 

SPMN module of SPFlow library implements the structure learning algorithm and 
claculates MEU on the learned structure for Sum-Product-Max Networks(**SPMN**)
which generalise Sum-Product Networks(**SPN**) for the class of decison-making problems.

## Getting Started

Use the forked version of SPFlow [https://github.com/c0derzer0/SPFlow] for installation.

## Using SPMN Module

#### Sample Dataset
Look at *spmn/data* folder for a list of sample datasets to use with spmn structure learning algorithm. 
*spmn/meta_data* contains information about 
*partial order, decision nodes, utility node, feature_names, meta_types* for each of the data sets.
```python
import pandas as pd    
csv_path = "Dataset5/Computer_diagnostician.tsv"
df = pd.DataFrame.from_csv(csv_path, sep='\t')
```
#### Provide initial parameters
Provide *partial order, decision nodes, utility node, feature_names, meta_types* for the dataset
```python
partial_order = [['System_State'], ['Rework_Decision'],
                 ['Logic_board_fail', 'IO_board_fail', 'Rework_Outcome', 
                 'Rework_Cost']]
utility_node = ['Rework_Cost']
decision_nodes = ['Rework_Decision']
feature_names = ['System_State', 'Rework_Decision', 'Logic_board_fail', 
                'IO_board_fail', 'Rework_Outcome', 'Rework_Cost']

from spn.structure.StatisticalTypes import MetaType
# Utility variable is the last variable. Other variables are of discrete type
meta_types = [MetaType.DISCRETE]*5+[MetaType.UTILITY]  
```
#### Pre-process data
This is not required if the data is a numpy ndarray ordered according to partial order
```python
from spn.algorithms.SPMNDataUtil import align_data
import numpy as np

df1, column_titles = align_data(df, partial_order)  # aligns data in partial order sequence
col_ind = column_titles.index(utility_node[0]) 

df_without_utility = df1.drop(df1.columns[col_ind], axis=1)
from sklearn.preprocessing import LabelEncoder
# transform categorical string values to categorical numerical values
df_without_utility_categorical = df_without_utility.apply(LabelEncoder().fit_transform)  
df_utility = df1.iloc[:, col_ind]
df = pd.concat([df_without_utility_categorical, df_utility], axis=1, sort=False)

train_data = df.values
```
#### Learn the structure of SPMN 

```python
from spn.algorithms.SPMN import learn_spmn
spmn = SPMN(partial_order , decision_nodes, utility_node, feature_names, 
            meta_types, cluster_by_curr_information_set=True,
            util_to_bin = False)
spmn_structure = spmn.learn_spmn(train_data)    
```
#### Plot the learned SPMN
```python
from spn.io.Graphics import plot_spn
plot_spn(spmn_structure, "computer_diagonistic.pdf", feature_labels=['SS', 'DD', 'LBF', 'IBF', 'RO', 'RC'])
```

    
#### Calculate Maximum Expected Utility from the learned SPMN
```python
from spn.algorithms.MEU import meu
test_data = [[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]
meu = meu(spmn_structure, test_data)
```    
The output for meu
```python
[242.90708442]
```
#### Additional functionality
We can convert utility variable to binary random variable using cooper transformation
```python  
from spn.algorithms.SPMNDataUtil import cooper_tranformation
# col_ind is index of utility variable in train data
train_data_with_bin_utility = cooper_tranformation(train_data, col_ind)   
spmn = SPMN(partial_order , decision_nodes, utility_node, feature_names, 
        meta_types, cluster_by_curr_information_set=True,
        util_to_bin = False)
spmn_structure = spmn.learn_spmn(train_data_with_bin_utility) 
```
## Papers implemented
* Mazen Melibari, Pascal Poupart, Prashant Doshi. "Sum-Product-Max Networks for Tractable Decision Making". In Proceedings of the Twenty-Fifth International Joint Conference on Artificial Intelligence, 2016.

    

