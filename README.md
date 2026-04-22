# Breast Cancer Tumor Classification

Binary classification of breast tumors as malignant or benign using logistic regression and decision trees, with emphasis on precision/recall tradeoffs, ROC analysis, and model interpretability.

## Overview

This project builds a clinical decision support model to classify breast tumors based on exam measurements of size and shape. The work covers the full ML pipeline: baseline modeling, regularization and hyperparameter tuning, evaluation with clinical metrics, and a comparison to interpretable shallow decision trees.

## Dataset

The [Breast Cancer Wisconsin dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) from scikit-learn contains 569 samples with 30 numeric features derived from digitized images of fine needle aspirate (FNA) biopsies.

**Target:** Binary — `1` (malignant) or `0` (benign)

## Methods

### Task 1 — Train/Test Split
Data split with at least 30% held out for testing, using `random_state=2024`.

### Task 2 — Random Baseline
A probabilistic baseline that predicts malignant with probability equal to the training set prevalence. Used to establish a lower-bound on meaningful performance.

| Metric | Baseline |
|--------|----------|
| Precision | ~56% |
| Recall | ~61% |

### Task 3 — Logistic Regression with Regularization
An unregularized logistic regression model (`penalty=None`) is trained, revealing mild overfitting. A train/validation split (30% of train) is used to select the L2 regularization parameter `C`, scanning values to reduce the train/validation accuracy gap without degrading overall accuracy. Final model uses `C=0.9`.

### Task 4 — Confusion Matrix & Clinical Metrics

| Metric | Regularized Model |
|--------|-------------------|
| True Positives | 100 |
| True Negatives | 62 |
| False Positives | 5 |
| False Negatives | 4 |
| **Precision** | **93%** |
| **Recall** | **96%** |

A dramatic improvement over the random baseline.

### Task 5 — ROC Curve
The ROC curve (AUC = 0.99) demonstrates near-perfect separation between classes. In a cancer detection context, a higher recall threshold is preferred — it is far more dangerous to miss a true malignancy (false negative) than to flag a benign tumor for follow-up (false positive).

### Task 6 — Shallow Decision Trees
Three decision tree classifiers are compared across `max_depth` values of 3, 5, and 7, evaluated on the validation set:

| max_depth | Accuracy | Precision | Recall |
|-----------|----------|-----------|--------|
| 3 | Best | Best | Best |
| 5 | Lower | Lower | Lower |
| 7 | Lower | Lower | Lower |

**Selected model: `max_depth=3`** — highest performance on all metrics and the simplest model (Occam's Razor). Shallow trees provide the added benefit of human interpretability: the decision path can be read and explained to a clinician.

## Key Findings

- Logistic regression with L2 regularization achieves 93% precision and 96% recall, far outperforming the baseline.
- The ROC curve (AUC = 0.99) indicates excellent discriminative ability.
- For cancer detection, **recall** is the more clinically important metric — missing a malignancy carries far greater cost than a false alarm.
- Shallow decision trees can match or approach logistic regression performance while being fully interpretable.

## Tech Stack

- Python 3
- scikit-learn (`LogisticRegression`, `DecisionTreeClassifier`, `ConfusionMatrixDisplay`, `RocCurveDisplay`, `precision_score`, `recall_score`, `accuracy_score`)
- NumPy, Matplotlib

## How to Run

```bash
pip install scikit-learn numpy matplotlib jupyter
jupyter notebook cancer.ipynb
```

No external data download needed — the dataset is loaded via scikit-learn.
