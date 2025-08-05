# Uncertainty-Aware Online Ensemble Transformer for Accurate Cloud Workload Forecasting in Predictive Auto-Scaling
This repository contains the official implementation of the paper *"Uncertainty-Aware Online Ensemble Transformer for Accurate Cloud Workload Forecasting in Predictive Auto-Scaling"*. The code provides a flexible framework for cloud workload forecasting tasks, supporting online learning, ensemble modeling, and various advanced time series forecasting architectures.


## Environment Setup
Install the required dependencies using the following command:
```bash
pip install -r requirements.txt
```
**Key Dependencies**:
- torch==1.7.1
- pandas==1.5.3
- numpy==1.23.5
- tqdm==4.66.5
- einops==0.6.0
- numexpr==2.8.6
- torchvision==0.8.2


## Quick Start
### Basic Forecasting Experiment
Run standard forecasting experiments on benchmark datasets (e.g., ETTh2) using:
```bash
bash run.sh
```
This script executes experiments with configurations including:
- Target dataset: `ETTh2_test`
- Supported models: `E3Former`, `onenet_fsnet`, `fsnet`, `fsnet_time`, `itransformer`, `timesnet`, `dlinear`
- Forecasting horizons: 60, 30, 10, 1
- Training epochs: 10
- Online learning mode: `full`


### Transfer Learning Experiment
Evaluate cross-dataset generalization with transfer learning tasks:
```bash
bash run_transfer.sh
```
This script focuses on transfer learning scenarios (e.g., ETT dataset transfer) with the same model suite and configurable prediction lengths.


## Core Configuration Parameters
| Parameter          | Description                                                                 | Default Value       |
|---------------------|-----------------------------------------------------------------------------|---------------------|
| `--data`            | Dataset name (supports `ETTh2_test`, `ETT_transfer`, etc.)                  | `ETTh2`             |
| `--method`          | Forecasting model selection                                                | `onenet_fsnet`      |
| `--seq_len`         | Input sequence length for encoder                                          | 1440                |
| `--pred_len`        | Prediction sequence length (forecasting horizon)                           | 1                   |
| `--train_epochs`    | Number of training epochs                                                  | 10                  |
| `--learning_rate`   | Optimizer learning rate                                                    | 1e-3                |
| `--online_learning` | Online learning strategy (currently supports `full`)                       | `full`              |
| `--seed`            | Random seed for reproducibility                                             | 2025                |


## Model Architecture
We compare performance of our proposed model `E3Former` with multiple state-of-the-art time series forecasting baselines, including:
- `onenet_fsnet`: Integrated feature selection and forecasting network
- `timesnet`: Time-series decomposition with Inception modules
- `itransformer`: Improved transformer for long sequence forecasting
- `dlinear`: Linear decomposition-based forecasting model


## Reproducibility
All experiments use fixed random seeds (configurable via `--seed`) and deterministic PyTorch settings to ensure result reproducibility. Adjust the `--itr` parameter to run multiple experiment iterations for statistical validation.