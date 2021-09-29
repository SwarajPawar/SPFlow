# SPMN 

SPMN module of SPFlow library implements the structure learning algorithm and 
claculates MEU on the learned structure for Sum-Product-Max Networks(**SPMN**)
which generalise Sum-Product Networks(**SPN**) for the class of decison-making problems.

## Getting Started

Use the **spmn** of SPFlow [https://github.com/SwarajPawar/SPFlow] for installation.

## Using SPMN Module

#### Sample Dataset
Look at *spmn/data* folder for a list of sample datasets to use with spmn structure learning algorithm. 
*spmn/meta_data* contains information about 
*partial order, decision nodes, utility node, feature_names, meta_types* for each of the data sets.
```python
import pandas as pd    
csv_path = "spn/data/Export_Textiles/Export_Textiles.tsv"
df = pd.DataFrame.from_csv(csv_path, sep='\t')
```
#### Provide initial parameters
Provide *partial order, decision nodes, utility node, feature_names, meta_types* for the dataset
```python
partial_order = [['Export_Decision'], ['Economical_State'], ['Profit']]
utility_node = ['Profit']
decision_nodes = ['Export_Decision']
feature_names = ['Export_Decision', 'Economical_State', 'Profit']

from spn.structure.StatisticalTypes import MetaType
# Utility variable is the last variable. Other variables are of discrete type
meta_types = [MetaType.DISCRETE]*2+[MetaType.UTILITY]  
```
#### Pre-process data
This is not required if the data is a numpy ndarray ordered according to partial order
```python
from spn.algorithms.SPMNDataUtil import align_data
import numpy as np

df, column_titles = align_data(df, partial_order)  # aligns data in partial order sequence
train_data = df.values
```
#### Learn the structure of SPMN 

```python
from spn.algorithms.SPMN import SPMN
spmn = SPMN(partial_order , decision_nodes, utility_node, feature_names, 
            meta_types, cluster_by_curr_information_set=True,
            util_to_bin = False)
spmn_structure = spmn.learn_spmn(train_data)    
```
#### Plot the learned SPMN
```python
from spn.io.Graphics import plot_spn
plot_spn(spmn_structure, "export_textiles.pdf", feature_labels=['ED', 'ES', 'Pr'])
```

    
#### Calculate Maximum Expected Utility from the learned SPMN
```python
from spn.algorithms.MEU import meu
test_data = [[np.nan, np.nan, np.nan]]
meu = meu(spmn_structure, test_data)
```    
The output for meu
```python
[1722313.8158882717]
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

    


