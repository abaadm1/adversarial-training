
# Adversarial Training for Robust Image Classification

PyTorch project that implements **adversarial training techniques** to defend neural networks against **FGSM** and **PGD attacks**. The pipeline covers **mixed adversarial training**, **loss averaging strategies**, **hyperparameter optimization**, and **robustness evaluation** on clean and perturbed data.

## Results

Final model performance on evaluation dataset:
- **Clean Accuracy**: 51.33% (meets 50% requirement)
- **FGSM Accuracy**: 39.90% (single-step attack robustness)
- **PGD Accuracy**: 38.07% (multi-step attack robustness)

Earlier iterations with complex regularization (label smoothing, heavy augmentation) failed to meet the 50% clean accuracy threshold. The final simple approach outperformed sophisticated techniques by 15%+ in clean accuracy.

## What this repo contains

| File | Purpose |
|------|--------|
| [`adversarial_training.ipynb`](adversarial_training.ipynb) | Main pipeline: custom dataset, FGSM/PGD attacks, ResNet-18 training, loss averaging, model saving. |
| [`requirements.txt`](requirements.txt) | Pinned Python dependencies. |
| [`.gitignore`](.gitignore) | Ignores model checkpoints, datasets, and Python artifacts. |

## Method

1. **Data Loading:** Custom `TaskDataset` handles RGB conversion and 32×32 resizing for consistent input preprocessing.
2. **Attacks:** FGSM (single-step) and PGD (3-step iterative) with epsilon=0.03 perturbation budget.
3. **Model:** ResNet-18 modified for 10-class classification, complying with competition constraints.
4. **Training:** Simultaneous optimization on clean + FGSM + PGD examples with uniform loss averaging.
5. **Key Insight:** Simple uniform averaging `(loss_clean + loss_fgsm + loss_pgd) / 3` outperformed complex weighting schemes.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux / macOS

pip install -r requirements.txt
```

## Running

Open `adversarial_training.ipynb` and run all cells. The notebook will:
1. Load and preprocess the training dataset
2. Initialize ResNet-18 model
3. Train with adversarial examples for 15 epochs
4. Save the final model as `robust_model.pt`

## Key Findings

- **Simplicity Wins**: Basic ResNet-18 with uniform loss averaging achieved best results
- **Overfitting Risk**: Training >20 epochs consistently reduced clean accuracy below 50%
- **Failed Approaches**: Label smoothing, heavy augmentation, complex loss weighting, learning rate scheduling
- **Trade-offs**: Balance between clean accuracy and adversarial robustness is crucial

## References (techniques cited in the original write-up)

- FGSM attack: [Goodfellow et al. 2015](https://arxiv.org/abs/1412.6572)
- PGD attack: [Madry et al. 2018](https://arxiv.org/abs/1706.06083)
- Adversarial training: [Goodfellow et al. 2015](https://arxiv.org/abs/1412.6572)
