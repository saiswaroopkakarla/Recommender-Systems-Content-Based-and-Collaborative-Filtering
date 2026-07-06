# Recommender Systems: Content-Based & Collaborative Filtering

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)
![SHAP](https://img.shields.io/badge/Explainability-SHAP%20%7C%20LIME%20%7C%20kNN-green)

A complete recommender systems assignment covering 6 parts and 13 tasks , from classical TF-IDF similarity to reinforcement learning agents and SHAP explainability,  implemented end-to-end in a single self-contained notebook.

**No dataset download required.** A synthetic MovieLens-style dataset (500 movies · 300 users · ~14K ratings) is auto-generated in-notebook.

---

## Results Summary

| Method | RMSE | Notes |
|---|---|---|
| **Hybrid Meta-Model (GBM)** | **0.5058** | Best overall — blends CBF + CF + popularity + user bias |
| User-Based CF (K=30) | 0.5377 | Strong memory-based baseline |
| User-Based CF (K=20) | 0.5861 | |
| Scipy Sparse SVD (k=50) | 0.6062 | Efficient truncated factorisation |
| Item-Based CF (K=20) | 0.7264 | Better memory efficiency than user-based |
| NumPy SVD (k=30) | 0.7289 | Full decomposition then truncated |
| Neural Network (CBF) | 0.9844 | Genre features alone insufficient |

**Key finding:** The GBM hybrid that blends CBF score, CF score, movie popularity, and user bias achieves the lowest RMSE — no single technique dominates across all user types.

---

## Coverage — 13 Tasks across 6 Parts

| Part | Task | Method |
|---|---|---|
| 1 — Content-Based Filtering | Task 1 | TF-IDF + Cosine Similarity |
| | Task 2 | Weighted User Profile (genre TF-IDF) |
| 2 — Collaborative Filtering | Task 3 | User-Based CF (Mean-Centred Cosine / Pearson) |
| | Task 4 | Item-Based CF (Cosine Similarity) |
| 3 — Matrix Factorisation | Task 5 | SVD from scratch (NumPy) |
| | Task 6 | Truncated SVD (SciPy Sparse — production-style) |
| 4 — Hybrid Model | Task 7 | Meta-Learning Hybrid (GradientBoosting on CBF + CF scores) |
| 5 — Learning-Based | Task 8 | Two-Branch Neural Network (Keras) |
| | Task 9 | RL: ε-Greedy MAB · UCB MAB · Q-Learning |
| 6 — Explainability | Task 10 | SHAP (TreeExplainer on GBM) |
| | Task 11 | k-NN Neighbourhood Explanations |
| | Task 12 | LIME (Model-Agnostic, Neural Net) |
| | Task 13 | Bias Analysis + Method Comparison |

---

## Repository Structure

```
.
├── recommender_systems_assignment.ipynb   # All 13 tasks, 30 cells
└── README.md
```

---

## Setup

```bash
git clone https://github.com/saiswaroopkakarla/Recommender-Systems-Content-Based-and-Collaborative-Filtering.git
cd Recommender-Systems-Content-Based-and-Collaborative-Filtering
pip install scikit-learn tensorflow shap lime
```

> **Note:** `scikit-surprise` is **not** used — it is incompatible with NumPy 2.x due to a Cython ABI mismatch. Task 6 uses `scipy.sparse.linalg.svds` instead (equivalent result, slightly better RMSE).

### Run

```bash
jupyter notebook recommender_systems_assignment.ipynb
# Kernel → Restart & Run All
```

The dataset is auto-generated in Cell 2. Nothing to download.

---

## Dataset

Synthetic MovieLens-style data (seed 42):

| Property | Value |
|---|---|
| Movies | 500 |
| Users | 300 |
| Ratings | ~14,257 |
| Rating scale | 1–5 |
| Genres | 16 binary flags |
| Matrix sparsity | ~90.5% |

> To use the real MovieLens 100K dataset instead, replace Cell 2 with:
> ```python
> import requests, zipfile
> from io import BytesIO
> r = requests.get('https://files.grouplens.org/datasets/movielens/ml-100k.zip')
> with zipfile.ZipFile(BytesIO(r.content)) as z:
>     z.extractall('.')
> ```

---

## Explainability Methods

| Method | Model explained | Tool |
|---|---|---|
| SHAP | GBM Hybrid Meta-Model | `shap.TreeExplainer` |
| k-NN Neighbourhoods | User-Based CF | Custom implementation |
| LIME | Two-Branch Neural Network | `lime.lime_tabular` |

---

## Notes

- All random seeds fixed at `42` for reproducibility
- RL agents use a restricted action space (200 movies, 20 users) for tractability
- Popularity bias ratio: **1.09×** — mild, below the 1.2 alert threshold
- Neural network training may vary slightly across runs due to TensorFlow CPU non-determinism

---

## Author

**Kakarla Sai Swaroop**  M25DE1023, IIT Jodhpur M.Tech Data Engineering  
Course: CSL7110  ML with Big Data
