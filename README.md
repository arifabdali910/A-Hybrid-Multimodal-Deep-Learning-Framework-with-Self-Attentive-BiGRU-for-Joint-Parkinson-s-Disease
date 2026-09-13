# Hybrid Multimodal Deep Learning Framework for Parkinson's Disease Staging and Depression Assessment

This folder contains the trained model checkpoints, evaluation outputs, and figures produced from a 5-fold cross-validation experiment on two classification tasks:

- **DEPRESSION_BINARY** — 2-class classification (e.g., depressed vs. not depressed)
- **DEPRESSION_SEVERITY** — 4-class classification (severity levels 0–3)

---

## 📁 Folder Contents

### Model Checkpoints
| File | Description |
|---|---|
| `model_fold1.pt` – `model_fold5.pt` | Trained model weights for each of the 5 cross-validation folds (~2.79 MB each, PyTorch format). |

### Summary Results
| File | Description |
|---|---|
| `results_summary.csv` | Mean ± std of all key metrics (Accuracy, F1, Precision, Recall, AUC-ROC, Kappa) across the 5 folds, for both tasks. |
| `classification_report_DEPRESSION_BINARY.csv` | Per-fold, per-class precision/recall/F1/support for the binary task. |
| `classification_report_DEPRESSION_SEVERITY.csv` | Per-fold, per-class precision/recall/F1/support for the severity task. |

### Figures
| File | Description |
|---|---|
| `accuracy_curves_DEPRESSION_BINARY.png` | Training/validation accuracy over epochs — binary task. |
| `accuracy_curves_DEPRESSION_SEVERITY.png` | Training/validation accuracy over epochs — severity task. |
| `loss_curves_all_folds.png` | Training/validation loss curves across all 5 folds. |
| `confusion_matrix_DEPRESSION_BINARY.png` | Confusion matrix — binary task. |
| `confusion_matrix_DEPRESSION_SEVERITY.png` | Confusion matrix — severity task. |
| `per_class_f1_DEPRESSION_BINARY.png` | Per-class F1 scores — binary task. |
| `per_class_f1_DEPRESSION_SEVERITY.png` | Per-class F1 scores — severity task. |
| `overall_metrics_comparison.png` | Side-by-side comparison of overall metrics between the two tasks. |
| `boxplot_metrics.png` | Distribution (spread across folds) of each metric. |
| `radar_chart.png` | Radar/spider plot comparing metrics across tasks. |
| `augmented_distributions.png` | Class distribution before/after data augmentation. |

---

## 📊 Key Results (Mean ± Std over 5 folds)

| Metric | DEPRESSION_SEVERITY | DEPRESSION_BINARY |
|---|---|---|
| Accuracy | 0.9985 ± 0.0020 | 0.9985 ± 0.0020 |
| F1 (Macro) | 0.9985 ± 0.0020 | 0.9980 ± 0.0027 |
| F1 (Weighted) | 0.9985 ± 0.0020 | 0.9985 ± 0.0020 |
| Precision | 0.9985 ± 0.0020 | 0.9990 ± 0.0013 |
| Recall | 0.9985 ± 0.0020 | 0.9970 ± 0.0040 |
| AUC-ROC | 1.0000 ± 0.0001 | 1.0000 ± 0.0001 |
| Cohen's Kappa | 0.9980 ± 0.0027 | 0.9959 ± 0.0054 |

**Note:** These are near-perfect scores across all folds. Before reporting or publishing these numbers, double-check for potential **data leakage** between train/validation/test splits (e.g., overlapping subjects, augmented copies of the same sample landing in both train and test sets), since real-world clinical classification tasks rarely achieve this level of accuracy.

---

## 🗂 Per-Fold Breakdown

Full per-class precision/recall/F1 values for every fold are available in:
- `classification_report_DEPRESSION_BINARY.csv`
- `classification_report_DEPRESSION_SEVERITY.csv`

Both tasks show consistent performance across all 5 folds, with Fold 1, 3, and 5 achieving perfect (1.00) scores on all classes, and Folds 2 and 4 showing minor dips (as low as 0.98 recall on class 0).

---

## 🔁 Reproducing / Loading the Models

```python
import torch

# Load a specific fold's model
model = torch.load("model_fold1.pt", map_location="cpu")
model.eval()

```
> Update the loading code above to match your model class definition if `torch.load` returns a state_dict rather than a full model object.
---

## 📌 Notes
- Metrics are averaged across the 5 folds; ± values represent standard deviation, not confidence intervals.
- "Avg_Accuracy_Raw" / "Avg_F1_Raw" columns in `results_summary.csv` are the unrounded raw averages before formatting.
