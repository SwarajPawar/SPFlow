
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


```
#### Set Dataset Context
Provide meta-type for the dataset variables
```python

```

#### Start the Anytime Learning for SPNs 

```python
import spn.algorithms.ASPMN

  
```
The aspmn.learn_aspmn() function generates and saves the plots for
for the SPNs at each iteration of the Anytime technique
These plots can be found at output_path/dataset




 
