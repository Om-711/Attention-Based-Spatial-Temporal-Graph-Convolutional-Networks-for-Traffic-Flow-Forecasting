# Attention-Based Spatial-Temporal Graph Convolutional Networks (ASTGCN) for Traffic Flow Forecasting

## Overview
This repository provides an implementation of the paper **"Attention-Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting"**. The model combines spatial and temporal dependencies in traffic networks using:
- **Graph Convolutional Networks (GCN)** to capture spatial correlations.
- **Attention Mechanisms** for adaptive importance weighting.
- **Temporal Convolutions** to extract time-series patterns.

## Features
- Implementation of ASTGCN using PyTorch
- Support for dynamic and static adjacency matrices
- Multi-step traffic flow forecasting
- Data preprocessing utilities for traffic datasets

## Installation
Clone this repository and install dependencies:
```bash
git clone https://github.com/your-repo/ASTGCN.git
cd ASTGCN
pip install -r requirements.txt
```

## Dataset
The model is designed for traffic datasets like **PEMS04** or **PEMS08**.

## Usage
### Training
To train the model, run:
```bash
python train.py --dataset metr-la --epochs 50 --batch_size 64 --lr 0.001
```
### Testing
To evaluate the trained model:
```bash
python test.py --dataset metr-la --checkpoint best_model.pth
```

## Model Architecture
1. **Spatial Attention Module** - Learns importance weights for different nodes in the graph.
2. **Temporal Attention Module** - Captures temporal correlations in time-series data.
3. **Graph Convolution Layer** - Uses adjacency matrices for spatial dependency modeling.
4. **Output Layer** - Predicts future traffic states.

## Results
The model achieves state-of-the-art performance on traffic forecasting tasks. Performance is measured using MAE, RMSE, and MAPE.

## References
- B. Guo, L. Lin, X. Feng, C. Song, and R. Xie, *"Attention-Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting"*.


