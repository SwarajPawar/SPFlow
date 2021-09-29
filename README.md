
# Anytime SPMNs:



The Anytime SPMNs module of the SPFlow library implements an anytime technique
to learn the SPMN models such that their performance (log-likelihood / MEU) 
improves given more time and nodes to learn the networks.

This anytime technique can learn SPMNs having performance comparable to the network
learned by the LearnSPMN algorithm using lower number nodes and in lesser learning times.

## Getting Started

Use the ![anytime_spmn](https://github.com/SwarajPawar/SPFlow/tree/anytime_spmn) branch of SPFlow for installation.

## Using Anytime SPN Module

#### Sample Dataset
Look at *src/spn/data* folder for a list of sample datasets to use with spmn structure learning algorithm. 


```python
import pandas as pd    

df = pd.read_csv(f"spn/data/{dataset}/{dataset}.tsv", sep='\t')
df, column_titles = align_data(df, partial_order)
data = df.values

```

#### Provide initial parameters
Provide *partial order, decision nodes, utility node, feature_names, meta_types* for the dataset
```python
partial_order = [['Export_Decision'], ['Economical_State'], ['Profit']]
utility_node = ['Profit']
decision_nodes = ['Export_Decision']
feature_names = ['Export_Decision', 'Economical_State', 'Profit']
feature_labels = ['ED', 'ES', 'Pr']
from spn.structure.StatisticalTypes import MetaType
# Utility variable is the last variable. Other variables are of discrete type
meta_types = [MetaType.DISCRETE]*2+[MetaType.UTILITY]  
```

#### Start the Anytime Learning for SPMNs 

```python
from spn.algorithms.ASPMN import Anytime_SPMN

anytime_spmn = Anytime_SPMN(dataset, path, partial_order , decision_nodes,
							 utility_node, feature_names, feature_labels, 	
							 meta_types, cluster_by_curr_information_set=True,
							 util_to_bin = False)
							 
for output in (aspmn.learn_aspmn(train, test, get_stats=True, evaluate_parallel=True)):

		spmn, stats = output

		#Plot the SPMN
		plot_spn(spmn, output_path/dataset/spmn.pdf', feature_labels=feature_labels)
  
```
The aspmn.learn_aspmn() function generates and saves the plots for
for the SPMNs at each iteration of the Anytime technique
These plots can be found at output_path/dataset

Note that the aspn.learn_aspmn() function is a generator function and generates 
the SPMN networks in a sequence and returns them at each iteration


 
