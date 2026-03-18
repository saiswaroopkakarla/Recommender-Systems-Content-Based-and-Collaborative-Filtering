# Recommender Systems: Content-Based and Collaborative Filtering

**Course:** ML with Big Data  
**Institute:** Indian Institute of Technology Jodhpur  M.Tech Data Engineering  
**Student:** Kakarla Sai Swaroop  
**Roll ID:** M25DE1023  

---

## Overview

A complete implementation of modern recommender system techniques, built end-to-end in a single self-contained Jupyter notebook. The notebook covers six parts and thirteen tasks — from classical TF-IDF similarity all the way to reinforcement learning agents and SHAP explainability.

The dataset is generated synthetically in-notebook (MovieLens-style: 500 movies · 300 users · ~14K ratings) so **no internet connection or manual download is required**.

---

## Repository Structure

```
.
├── recommender_systems_assignment.ipynb   # Main notebook (all 13 tasks)
├── README.md                              # This file
```

---

## Parts & Tasks

| Part | Task | Method |
|------|------|--------|
| 1 — Content-Based Filtering | Task 1 | TF-IDF + Cosine Similarity |
| | Task 2 | Weighted User Profile (genre TF-IDF) |
| 2 — Collaborative Filtering | Task 3 | User-Based CF (Pearson / Mean-Centred Cosine) |
| | Task 4 | Item-Based CF (Cosine Similarity) |
| 3 — Matrix Factorisation | Task 5 | SVD from scratch (NumPy) |
| | Task 6 | Truncated SVD (SciPy Sparse — production-style) |
| 4 — Hybrid Model | Task 7 | Meta-Learning Hybrid (GradientBoosting on CBF + CF scores) |
| 5 — Learning-Based | Task 8 | Two-Branch Neural Network (Keras) |
| | Task 9 | RL: ε-Greedy MAB · UCB MAB · Q-Learning |
| 6 — Explainability | Task 10 | SHAP (Feature-Based) |
| | Task 11 | k-NN Neighbourhood Explanations |
| | Task 12 | LIME (Model-Agnostic, Neural Net) |
| | Task 13 | Bias Analysis + Method Comparison |

---

## Setup & Requirements

### Python Version
Python 3.9 or later recommended.

### Install Dependencies

Run the first cell in the notebook, or install manually:

```bash
pip install scikit-learn tensorflow shap lime
```

> **Note:** `scikit-surprise` is **not** used in this notebook — it is incompatible with NumPy 2.x due to a Cython ABI mismatch. Task 6 uses `scipy.sparse.linalg.svds` instead, which achieves an equivalent (and slightly better) result.

### Run the Notebook

```bash
jupyter notebook recommender_systems_assignment.ipynb
```

Then in Jupyter: **Kernel → Restart & Run All**

The dataset is auto-generated in Cell 2. No files need to be downloaded.

---

## Results Summary

| Method | RMSE | Notes |
|--------|------|-------|
| Hybrid Meta-Model (GBM) | **0.5058** | Best overall |
| User-Based CF (K=30) | 0.5377 | Strong memory-based baseline |
| User-Based CF (K=20) | 0.5861 | |
| Scipy Sparse SVD (k=50) | 0.6062 | Efficient truncated factorisation |
| Item-Based CF (K=20) | 0.7264 | Better memory efficiency in production |
| NumPy SVD (k=30) | 0.7289 | Full decomposition, then truncated |
| Neural Network (CBF) | 0.9844 | Genre features alone insufficient |

**Key finding:** The GBM hybrid that blends CBF score, CF score, movie popularity, and user bias achieves the lowest RMSE — validating that no single technique dominates across all user types.

---

## Dataset Details

Synthetic MovieLens-style data generated with `numpy.random.seed(42)`:

| Property | Value |
|----------|-------|
| Movies | 500 |
| Users | 300 |
| Ratings | ~14,257 (after dedup) |
| Rating scale | 1–5 integer |
| Genres | 16 binary flags (Action, Drama, Sci-Fi, etc.) |
| Release years | 1970–2019 |
| Matrix sparsity | ~90.5% |

> To swap in the real MovieLens 100K dataset, replace Cell 2 with:
> ```python
> import requests, zipfile
> from io import BytesIO
> r = requests.get('https://files.grouplens.org/datasets/movielens/ml-100k.zip')
> with zipfile.ZipFile(BytesIO(r.content)) as z:
>     z.extractall('.')
> ```
> Then adjust the column names to match `u.data` and `u.item` file formats.

---

## Explainability Methods

| Method | Model Explained | Tool |
|--------|----------------|------|
| SHAP | GBM Hybrid Meta-Model | `shap.TreeExplainer` |
| k-NN Neighbourhoods | User-Based CF | Custom implementation |
| LIME | Two-Branch Neural Network | `lime.lime_tabular` |

---

## Notes

- All random seeds fixed at `42` for reproducibility.
- Neural network training may show slight variation across runs due to TensorFlow non-determinism on CPU.
- The RL agents (MAB, Q-Learning) use a restricted action space (200 movies, 20 users) for tractability.
- Popularity bias ratio measured at **1.09×** — mild bias, below the 1.2 alert threshold.
