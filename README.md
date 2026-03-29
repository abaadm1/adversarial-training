
# Adversarial Training for Robust Image Classification

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)

A comprehensive implementation of adversarial training techniques to defend neural networks against FGSM and PGD attacks. This project demonstrates practical robustness in image classification using PyTorch.

## 🎯 Project Overview

In this project, we developed a robust image classifier to defend against adversarial attacks, specifically **FGSM** (Fast Gradient Sign Method) and **PGD** (Projected Gradient Descent). The model was evaluated on both clean and adversarial data, achieving balanced performance across all scenarios.

### 📊 Performance Results

| Metric | Accuracy | Status |
|--------|----------|--------|
| **Clean Accuracy** | 51.33% | ✅ Meets ≥50% requirement |
| **FGSM Accuracy** | 39.90% | Strong robustness |
| **PGD Accuracy** | 38.07% | Effective defense |

---

## 🛠️ Technical Stack

- **Framework**: PyTorch
- **Model**: ResNet-18 (modified for 10-class classification)
- **Attacks**: FGSM, PGD
- **Training**: Adversarial training with mixed inputs
- **Dataset**: Custom image dataset (32×32 RGB)

---

## 📁 Repository Structure

```
adversarial-training/
├── README.md                    # Project documentation
├── adversarial_training.ipynb   # Main implementation
├── requirements.txt             # Python dependencies
└── robust_model.pt             # Trained model checkpoint
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Jupyter Notebook

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/adversarial-training.git
cd adversarial-training
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch the notebook:
```bash
jupyter notebook adversarial_training.ipynb
```

---

## 🧠 Methodology

### Adversarial Training Strategy

Our approach combines three types of training data in each batch:

1. **Clean Images**: Original training samples
2. **FGSM Examples**: Single-step adversarial perturbations
3. **PGD Examples**: Multi-step iterative attacks

### Loss Function

```python
loss = (loss_clean + loss_fgsm + loss_pgd) / 3
```

This uniform averaging provides the best balance between clean accuracy and adversarial robustness.

### Attack Implementations

#### FGSM Attack
```python
def fgsm_attack(model, images, labels, epsilon):
    images.requires_grad = True
    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    model.zero_grad()
    loss.backward()
    return torch.clamp(images + epsilon * images.grad.sign(), 0, 1)
```

#### PGD Attack
```python
def pgd_attack(model, images, labels, epsilon=0.03, alpha=0.01, steps=3):
    ori = images.clone().detach()
    images = ori + 0.001 * torch.randn_like(ori)
    for _ in range(steps):
        images.requires_grad = True
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad()
        loss.backward()
        images = images + alpha * images.grad.sign()
        eta = torch.clamp(images - ori, min=-epsilon, max=epsilon)
        images = torch.clamp(ori + eta, min=0, max=1).detach()
    return images
```

---

## ⚙️ Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Epochs** | 15 | Optimal to prevent overfitting |
| **Batch Size** | 32 | Standard for image classification |
| **Optimizer** | Adam | Adaptive learning rate |
| **Learning Rate** | 0.001 | Conservative for stability |
| **Epsilon** | 0.03 | Max perturbation strength |
| **PGD Steps** | 3 | Computationally efficient |

---

## 🔬 Key Findings

### What Worked Best
- **Simple Approach**: Basic ResNet-18 with minimal modifications
- **Uniform Loss Averaging**: Equal weighting of all three loss components
- **Conservative Training**: 15 epochs prevented overfitting to adversarial examples

### Lessons Learned
- **Overfitting Risk**: Training >20 epochs harms clean accuracy
- **Simplicity Wins**: Complex regularization techniques reduced performance
- **Trade-offs**: Balance between clean accuracy and robustness is crucial

### Failed Experiments
- Label smoothing and heavy data augmentation
- Complex loss weighting schemes
- Learning rate scheduling

---

## 📈 Performance Analysis

### Training Duration Impact

| Epochs | Clean Accuracy | Evaluation Status |
|--------|----------------|-------------------|
| **10** | ≥ 50% | ✅ Accepted |
| **15** | 51.33% | ✅ Accepted (Final) |
| **20** | ~50% | ✅ Accepted |
| **30+** | < 50% | ❌ Rejected |

**Key Insight**: Overfitting to adversarial examples harms generalization on clean data.

---

## 🎯 Conclusion

This project demonstrates that **simplicity in adversarial training outperforms complex regularization techniques**. The final model achieves a robust balance between clean accuracy and adversarial defense while maintaining computational efficiency.

The implementation provides a solid foundation for understanding practical adversarial machine learning and serves as a reference for robust neural network development.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
