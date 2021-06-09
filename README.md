
# Anytime SPNs:

SPFlow, an open-source Python library providing a simple interface to inference,
learning  and  manipulation  routines  for  deep  and  tractable  probabilistic  models called Sum-Product Networks (SPNs).
The library allows one to quickly create SPNs both from data and through a domain specific language (DSL).
It efficiently implements several probabilistic inference routines like computing marginals, conditionals and (approximate) most probable explanations (MPEs)
along with sampling as well as utilities for serializing,plotting and structure statistics on an SPN.

The Anytime SPNs module of the SPFlow library implements an anytime technique
to learn the SPN models such that their performance (log-likelihood) increases
until convergence, given more time and nodes to learn the networks.

This anytime technique can learn SPNs having performance comparable to the network
learned by the LearnSPN algorithm using lower number nodes and in lesser learning times.

## Getting Started

Use the anytime_spn branch of SPFlow [https://github.com/SwarajPawar/SPFlow] for installation.

## Using Anytime SPN Module

#### Sample Dataset
Look at *src/spn/data* folder for a list of sample datasets to use with spn structure learning algorithm. 


```python
import pandas as pd    

df = pd.read_csv(f"spn/data/binary/nltcs.ts.data", sep=',')
train_data = df.values
df = pd.read_csv(f"spn/data/binary/nltcs.test.data", sep=',')
test_data = df.values
```
#### Set Dataset Context
Provide meta-type for the dataset variables
```python
var = data.shape[1]
ds_context = Context(meta_types=[MetaType.DISCRETE]*var)
ds_context.add_domains(data)
```

#### Start the Anytime Learning for SPNs 

```python
import spn.algorithms.ASPN

aspn = ASPN('nltcs', 'output_path', ds_context)
spn_structure, stats = aspn.learn_aspn(train_data, test_data)    
```
