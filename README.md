# Hybrid CNN-SNN Breast Cancer Histopathology Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.9+](https://img.shields.io/badge/pytorch-1.9+-red.svg)](https://pytorch.org/)
[![SpikingJelly](https://img.shields.io/badge/SpikingJelly-0.0.0.0.12-orange.svg)](https://github.com/fangwei123456/spikingjelly)

## 📋 Overview

This repository contains the implementation of a novel hybrid architecture combining Convolutional Neural Networks (CNNs) with Spiking Neural Networks (SNNs) for breast cancer histopathology image classification. The model leverages EfficientNet-B3 as a feature extractor and integrates temporal SNN layers for classification, achieving state-of-the-art performance on the BreakHis dataset.

### Key Features:
- **Hybrid Architecture**: Combines EfficientNet feature extraction with temporal SNN processing
- **High Accuracy**: Achieves 99.70% accuracy and 1.0000 ROC AUC on BreakHis dataset
- **Computational Efficiency**: 27% reduction in FLOPs compared to CNN-only models
- **Enhanced Interpretability**: Temporal spike pattern analysis for feature importance
- **Extensive Evaluation**: Comprehensive performance metrics and comparative analysis

## 📊 Results

Our hybrid CNN-SNN model achieved the following performance metrics on the BreakHis dataset at 400× magnification:

| Metric               | Value   |
|---------------------|---------|
| Accuracy            | 0.9970  |
| Precision           | 0.9986  |
| Recall (Sensitivity) | 0.9962  |
| F1-Score            | 0.9974  |
| Specificity         | 0.9981  |
| Balanced Accuracy   | 0.9971  |
| MCC                 | 0.9939  |
| Cohen's Kappa       | 0.9939  |
| Log Loss            | 0.0120  |
| ROC AUC             | 1.0000  |
| Youden's J          | 0.9943  |
| F2-Score            | 0.9967  |

## 🔧 Requirements

### Environment
- Python 3.8+
- PyTorch 1.9.0+
- CUDA 11.1+ (for GPU acceleration)
- Google Colab Pro (recommended for reproduction)

### Dependencies
```
pytorch==1.9.0
torchvision==0.10.0
spikingjelly==0.0.0.0.12
scikit-learn==1.0.1
opencv-python==4.5.4
numpy==1.21.5
pandas==1.3.5
matplotlib==3.5.0
efficientnet-pytorch==0.7.1
ptflops==0.6.9
```

Install dependencies using:
```bash
pip install -r requirements.txt
```

## 📁 Repository Structure

```
├── data/                         # Data handling utilities
│   ├── dataset.py                # Dataset class and data loading functions
│   └── preprocessing.py          # Image preprocessing functions
├── models/                       # Model definitions
│   ├── cnn_backbone.py           # EfficientNet backbone implementation
│   ├── snn_layers.py             # Spiking neural network layer implementations
│   ├── temporal_encoding.py      # Temporal encoding strategies
│   └── hybrid_model.py           # Complete hybrid CNN-SNN architecture
├── utils/                        # Utility functions
│   ├── visualization.py          # Visualization tools (Grad-CAM, spike patterns)
│   ├── metrics.py                # Performance metrics calculation
│   └── training.py               # Training utilities and callbacks
├── experiments/                  # Experiment configurations
│   ├── baseline_models.py        # Baseline model configurations
│   ├── ablation_studies.py       # Ablation study configurations
│   └── hyperparameter_tuning.py  # Hyperparameter search space
├── notebooks/                    # Jupyter notebooks
│   ├── data_exploration.ipynb    # Dataset exploration and visualization
│   ├── model_training.ipynb      # Main training notebook
│   ├── results_analysis.ipynb    # Performance analysis and visualization
│   └── grad_cam_visualization.ipynb # Feature visualization with Grad-CAM
├── scripts/                      # Executable scripts
│   ├── download_dataset.py       # Script to download and prepare the BreakHis dataset
│   ├── train.py                  # Training script
│   └── evaluate.py               # Evaluation script
├── configs/                      # Configuration files
│   ├── model_config.yaml         # Model architecture configuration
│   └── training_config.yaml      # Training hyperparameters
├── requirements.txt              # Package dependencies
├── LICENSE                       # MIT License
└── README.md                     # Repository documentation
```

## 🚀 Getting Started

### Dataset Preparation

1. Download the BreakHis dataset from [Mendeley Data](https://data.mendeley.com/datasets/ywsbh3ndr8/5)
2. Extract the dataset to a local directory
3. Run the dataset preparation script:
```bash
python scripts/download_dataset.py --data_path /path/to/breakhis/dataset --output_dir ./data/processed
```

### Training the Model

Run the training script:
```bash
python scripts/train.py --config configs/training_config.yaml --data_dir ./data/processed --output_dir ./outputs
```

Alternatively, you can use the Google Colab notebook:
```bash
jupyter notebook notebooks/model_training.ipynb
```

### Evaluation

Evaluate a trained model:
```bash
python scripts/evaluate.py --model_path ./outputs/best_model.pth --data_dir ./data/processed --output_dir ./results
```

## 📓 Notebook Details

The main notebook (`model_training.ipynb`) contains the following sections:

1. **Environment Setup**
   - Package installation
   - Configuration setup
   - GPU availability check

2. **Data Loading and Preprocessing**
   - Dataset download and extraction
   - Image preprocessing pipeline
   - Data augmentation implementation
   - Dataset splitting (train/val/test)

3. **Model Architecture**
   - EfficientNet backbone configuration
   - Temporal encoding layer implementation
   - SNN layer implementation
   - Hybrid model integration

4. **Training Process**
   - CNN pre-training phase
   - End-to-end hybrid training
   - Learning rate scheduling
   - Surrogate gradient method implementation

5. **Evaluation and Analysis**
   - Performance metrics calculation
   - Comparative analysis with baselines
   - Confusion matrix and ROC curve visualization
   - Computational efficiency measurement

6. **Feature Visualization**
   - Grad-CAM implementation
   - Spike pattern visualization
   - Feature importance analysis

7. **Ablation Studies**
   - Component-wise performance analysis
   - Hyperparameter sensitivity analysis

## 🔍 Model Architecture

### EfficientNet Backbone
- EfficientNet-B3 pre-trained on ImageNet
- Compound scaling with depth factor 1.4 and width factor 1.2
- Feature output dimension: 1,536

### Temporal Encoding Layer
- Normalization of CNN feature maps to [0,1]
- Probabilistic spike generation over 20 time steps
- Rate-based encoding preserving feature importance through spike density

### Spiking Neural Network
- Hidden layer: 256 Leaky Integrate-and-Fire (LIF) neurons
- Output layer: 2 LIF neurons (benign/malignant)
- Membrane leak factor (λ): 0.95
- Threshold potential: 1.0
- Arc-tangent surrogate gradient function

## 🏆 Comparative Analysis

Performance comparison with other state-of-the-art methods:

| Model                    | Accuracy | F1-Score | AUC    |
|--------------------------|----------|----------|--------|
| Our Hybrid CNN-SNN       | 0.9970   | 0.9974   | 1.0000 |
| CNN-LSTM (Kaddes et al.) | 0.9857   | 0.9862   | 0.9973 |
| Class-Wise Mean Teacher  | 0.9690   | 0.9708   | 0.9901 |
| Ensemble CNN             | 0.9500   | 0.9523   | 0.9812 |
| VGG19-SVM                | 0.9321   | 0.9334   | 0.9698 |
| DenseNet-169             | 0.9519   | 0.9532   | 0.9801 |

## 🧠 Inference

```python
# Example inference code
import torch
from models.hybrid_model import HybridCNNSNN

# Load model
model = HybridCNNSNN()
model.load_state_dict(torch.load('outputs/best_model.pth'))
model.eval()

# Preprocess image
from data.preprocessing import preprocess_image
image = preprocess_image('path/to/image.png')

# Inference
with torch.no_grad():
    output = model(image.unsqueeze(0))
    prediction = torch.argmax(output, dim=1).item()
    
# Map prediction to class
class_names = ['Benign', 'Malignant']
print(f"Predicted class: {class_names[prediction]}")
```

## 📝 Training Protocol

The model is trained in two phases:

### Phase 1: CNN Pre-training
- 30 epochs
- Learning rate: 1e-4 with cosine decay
- Weight decay: 1e-5
- Loss function: Cross-entropy
- Batch size: 32
- Early stopping patience: 10 epochs

### Phase 2: End-to-End Hybrid Training
- 50 epochs
- Learning rate: 1e-4 with cosine decay
- Weight decay: 1e-5
- Surrogate gradient function: Arc-tangent approximation
- SNN time steps: 20
- Batch size: 32
- Early stopping patience: 10 epochs

## 🎯 Hyperparameter Optimization

Key hyperparameters and their optimal values:

| Hyperparameter           | Optimal Value |
|--------------------------|---------------|
| SNN Leak Factor          | 0.95          |
| SNN Time Steps           | 20            |
| SNN Hidden Neurons       | 256           |
| Learning Rate            | 1e-4          |
| Weight Decay             | 1e-5          |
| Batch Size               | 32            |

## 🔬 Ablation Studies

Impact of different components on performance:

| Configuration                | Accuracy | F1-Score | ROC AUC |
|-----------------------------|----------|----------|---------|
| Full hybrid CNN-SNN          | 0.9970   | 0.9974   | 1.0000  |
| Without temporal encoding    | 0.9866   | 0.9871   | 0.9975  |
| Single SNN layer             | 0.9899   | 0.9907   | 0.9983  |
| Reduced time steps (T=10)    | 0.9933   | 0.9938   | 0.9991  |
| Increased time steps (T=30)  | 0.9966   | 0.9972   | 0.9998  |

## 📊 Visualization Tools

The repository includes several visualization tools:

### Grad-CAM
```python
from utils.visualization import generate_gradcam

# Generate Grad-CAM for a specific image
gradcam_map = generate_gradcam(model, image, target_layer='features.blocks.11')
```

### Spike Pattern Visualization
```python
from utils.visualization import visualize_spike_patterns

# Visualize spike patterns for a specific image
spike_patterns = visualize_spike_patterns(model, image)
```

### Performance Metrics Visualization
```python
from utils.visualization import plot_roc_curve, plot_confusion_matrix

# Plot ROC curve
plot_roc_curve(y_true, y_pred_prob)

# Plot confusion matrix
plot_confusion_matrix(y_true, y_pred)
```

## 📚 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{hybrid_cnn_snn_2025,
  title={Hybrid CNN-Spiking Neural Network for Improved Breast Cancer Histopathology Classification},
  author={Author, A. and Author, B. and Author, C.},
  journal={Journal Name},
  year={2025},
  volume={},
  pages={},
  doi={}
}
```

## 🔗 Related Works

- Kaddes, M., Ayid, Y.M., Elshewey, A.M., et al. Breast cancer classification based on hybrid CNN with LSTM model. Sci Rep 15, 4409 (2025). https://doi.org/10.1038/s41598-025-88459-6
- Dutta A, Ghosh B, Paul A, et al. Deep Learning in the Treatment of Cancer: A Review on Histopathological Image Processing. Cancers. 2023;15(7):7717712. https://onlinelibrary.wiley.com/doi/10.1155/2023/7717712
- Spanhol FA, Oliveira LS, Petitjean C, Heutte L. A dataset for breast cancer histopathological image classification. IEEE Trans Biomed Eng. 2016;63(7):1455-1462. https://data.mendeley.com/datasets/ywsbh3ndr8/5

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📬 Contact

For questions or feedback, please open an issue.
