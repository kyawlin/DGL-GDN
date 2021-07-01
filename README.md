# DGL-GDN
Deep Graph Library (DGL) port of  [Graph Neural Network-Based Anomaly Detection in Multivariate Time Series(AAAI'21)](https://www.aaai.org/AAAI21Papers/AAAI-5076.DengA.pdf)
Requirements
  * [DGL](https://www.dgl.ai/)
  
[Original implementation](https://github.com/d-ailin/GDN/blob/main/README.md) in Pytorch-Geometric.

It detects anomalies in multivariate time seires and have four steps
1. Sensor Embedding
2. Learning Graph Structure
3. Using Graph Attention Network to forecast
4. Calculate Deviation Scores to identify anomalous events.

Dataset is  expert-labeled telemetry
anomaly datafrom Mars Science Laboratory [MSL](https://arxiv.org/pdf/1802.04431.pdf).


