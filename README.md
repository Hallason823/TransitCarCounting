# Transit Car Counting
# Transit Car Counting

Research project for classifying traffic state on a scale model intersection using transfer learning. A top-down camera captures the model, and the classifier predicts the occupancy class based on the number of cars visible on the **right** and **bottom** road lanes.

## Class Mapping

| Class | Right lane | Bottom lane |
|-------|------------|-------------|
| 0     | 0          | 0           |
| 1     | 1          | 0           |
| 2     | 0          | 1           |
| 3     | 1          | 1           |
| 4     | 2          | 0           |
| 5     | 0          | 2           |
| 6     | 2          | 1           |
| 7     | 1          | 2           |
| 8     | 2          | 2           |

## Repository Structure

```text
TransitCarCounting/
├── docs/                          # Academic references
├── images/
├── src/
│   ├── main.py                    # Main execution script
│   ├── taskManager.py
│   ├── config/
│   ├── data/
│   ├── models/                    # ResNetCounter, EfficientNetCounter + Training loop
│   ├── classifier/                # Accuracy / F1 metrics       
│   └── plot/
├── README.md
└── requirements.txt
```

### `/images` — Dataset

Place your dataset here. Two layouts are supported:

- **Subfolders by count**: `images/0/`, `images/1/`, `images/3/`, …
- **CSV annotations**: `images/annotations.csv` with columns `path` and `count`

### `/src` — Source Code

Two pretrained models adapted for car counting:

- **RESNET** — ResNet18 (ImageNet pretrained) with a classification head
- **EFFICIENTNET** — EfficientNet-B0 (ImageNet pretrained) with a classification head

## Installation (setup)

```bash
pip install -r requirements.txt
```

## How to Run

### RESNET (ResNet18)

```bash
python src/main.py RESNET --num_iter=1 --split_mode=Hold-out --num_epochs=30 --batch_size=32 --learning_rate=0.0001 --weights=0.7,0.1
```

### EFFICIENTNET (EfficientNet-B0)

```bash
python src/main.py EFFICIENTNET --num_iter=1 --split_mode=Hold-out --num_epochs=30 --batch_size=32 --learning_rate=0.0001 --weights=0.7,0.1
```

### K-fold cross-validation

```bash
python src/main.py RESNET --num_iter=1 --split_mode=K-fold cross-validation --num_epochs=30 --k_groups=5 --weights=0.7,0.1
```

### Unfreeze backbone (full fine-tuning)

```bash
python src/main.py EFFICIENTNET --num_epochs=50 --freeze_backbone=False --learning_rate=0.00001
```

## Parameters

| Parameter           | Description                              | Default  |
|---------------------|------------------------------------------|----------|
| `--num_iter`        | Number of full experiment runs           | 1        |
| `--split_mode`      | `Hold-out` or `K-fold cross-validation`  | Hold-out |
| `--num_epochs`      | Training epochs                          | 30       |
| `--batch_size`      | Batch size                               | 32       |
| `--learning_rate`   | Optimizer learning rate                  | 0.0001   |
| `--weights`         | Train/val proportions, e.g. `0.7,0.1`    | 0.7,0.1  |
| `--k_groups`        | K for k-fold cross-validation            | 5        |
| `--freeze_backbone` | Freeze ImageNet weights (`True`/`False`) | True     |

## Models

### 1. ResNet18 (RESNET)

Classic residual network pretrained on ImageNet.

- Backbone frozen by default (only the classification head trains)
- Embedding size: 512 → 256 → num_classes
- Optimizer: Adam | Loss: CrossEntropyLoss

### 2. EfficientNet-B0 (EFFICIENTNET)

Efficient compound-scaled network pretrained on ImageNet.

- Backbone frozen by default
- Embedding size: 1280 → 256 → num_classes
- Optimizer: Adam | Loss: CrossEntropyLoss

## Outputs

| Path                                           | Content                                           |
|------------------------------------------------|---------------------------------------------------|
| `src/plot/results_images/`                     | Training loss and accuracy curves (PNG)           |
| `src/plot/confusion_matrix_images/`            | Confusion matrix per run (PNG)                    |
| `src/models/checkpoints/RESNET_best.pth`       | Best ResNet18 checkpoint (lowest val loss)        |
| `src/models/checkpoints/EFFICIENTNET_best.pth` | Best EfficientNet-B0 checkpoint (lowest val loss) |

## Metrics

- **Accuracy** — overall correct predictions
- **F1-Score (macro)** — unweighted mean F1 across all count classes
- **F1-Score (weighted)** — F1 weighted by class support

## Technologies

- Python
- PyTorch/TorchVision
- NumPy
- scikit-learn
- Matplotlib/Seaborn
- Pillow
