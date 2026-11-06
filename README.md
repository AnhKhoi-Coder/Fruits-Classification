
# Fruits-360 Image Classification

Deep Learning project comparing CNN Baseline and FasterNet-T1 for fruit image classification using the Fruits-360 dataset.

## 1. Project Overview

This project implements and evaluates two deep learning models:

- CNN Baseline
- FasterNet-T1

The main objectives are:
- Image classification on the Fruits-360 dataset.
- Comparison of classification performance.
- Evaluation of model complexity.
- Benchmarking inference latency and throughput.

## 2. Dataset

**Dataset:** Fruits-360

**Source:** [Fruits-360 on Kaggle](https://www.kaggle.com/datasets/moltean/fruits)

The dataset is not included in this repository because of its large size.

Download the dataset from Kaggle and prepare the directory structure required by the training scripts.

## 3. Models

### CNN Baseline

A custom Convolutional Neural Network used as the baseline model.

### FasterNet-T1

A lightweight neural network architecture using Partial Convolution (PConv).

Original paper: https://arxiv.org/abs/2303.03667

Official implementation: https://github.com/JierunChen/FasterNet

## 4. Project Structure

```text
Fruits-Classification/
│
├── .vscode/
│   └── settings.json
│
├── configs/
│
├── FasterNet/
│   ├── cfg/
│   │   ├── fasternet_l.yaml
│   │   ├── fasternet_m.yaml
│   │   ├── fasternet_s.yaml
│   │   ├── fasternet_t0.yaml
│   │   ├── fasternet_t1.yaml
│   │   └── fasternet_t2.yaml
│   │
│   ├── data/
│   │   ├── custom_imagenet_data.py
│   │   ├── data_api.py
│   │   └── __init__.py
│   │
│   ├── detection/
│   │   ├── backbones/
│   │   ├── configs/
│   │   ├── benchmark.py
│   │   ├── get_flops.py
│   │   ├── test.py
│   │   ├── train.py
│   │   ├── dist_test.sh
│   │   └── dist_train.sh
│   │
│   ├── models/
│   │   ├── fasternet.py
│   │   ├── model_api.py
│   │   ├── registry.py
│   │   └── __init__.py
│   │
│   ├── utils/
│   │   ├── fuse_conv_bn.py
│   │   ├── loss.py
│   │   ├── utils.py
│   │   └── __init__.py
│   │
│   ├── README.MD
│   ├── requirements.txt
│   └── train_test.py
│
├── models/
│   └── cnn_baseline.py
│
├── results/
│   ├── training_plots/
│   │   ├── 01_CNN_Accuracy.png
│   │   ├── 02_CNN_Loss.png
│   │   ├── 03_FasterNet_Accuracy.png
│   │   └── 04_FasterNet_Loss.png
│   │
│   ├── cnn_baseline_classification_report.csv
│   ├── cnn_baseline_confusion_matrix.csv
│   ├── cnn_baseline_history.csv
│   ├── fasternet_t1_classification_report.csv
│   ├── fasternet_t1_confusion_matrix.csv
│   ├── fasternet_t1_history.csv
│   └── inference_benchmark.csv
│
├── scripts/
│   ├── benchmark_models.py
│   ├── evaluate_cnn.py
│   ├── evaluate_fasternet.py
│   ├── train_cnn.py
│   └── train_fasternet.py
│
├── utils/
│
├── check_dataset.py
├── check_label_mapping.py
├── dataset.py
├── fix_class_name.py
├── plot_training_history.py
│
├── .gitignore
└── README.md
```

## 5. Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd Fruits-Classification
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 6. Dataset Preparation

Download Fruits-360 from Kaggle.

Extract the dataset and configure the dataset path according to the scripts in this project.

## 7. Training and Evaluation

Please refer to the scripts directory for the training and evaluation implementations.

## 8. Experimental Results

Training histories, evaluation results, and inference benchmark results are available in the results directory.

## 9. References

- Fruits-360 Dataset: https://www.kaggle.com/datasets/moltean/fruits
- FasterNet Paper: https://arxiv.org/abs/2303.03667
- FasterNet Official Repository: https://github.com/JierunChen/FasterNet
