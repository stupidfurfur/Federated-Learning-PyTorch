## Comparison of Federated Learning Global Model Aggregation Algorithms

### Abstract

  This study investigates the impact of different aggregation algorithms on Federated Learning (FL) in scenarios involving non-IID datasets. 
  FL enables collaborative model training while preserving data privacy, making it particularly relevant for sectors like healthcare. 
  How ever, the non-IID nature of client data can lead to challenges such as overfitting and reduced model accuracy. 
  We compare four aggregation algorithms FedAvg, ABAVG, ACC inverse, and IDA using non-IID datasets with varying label distributions. 
  The MNIST dataset is used for testing, and both MLP and CNN models are trained. 
  Results show that ABAVG achieves the highest accuracy (91.02%) in the MLP model, while ACC inverse, though having the lowest loss, demonstrates overfitting with lower accuracy. 
  In the CNN model, Salmeron et al. method yields the best accuracy (97.04%), despite higher loss. 
  These findings high light the importance of choosing appropriate aggregation strategies to optimize model performance in non-IID environments.

