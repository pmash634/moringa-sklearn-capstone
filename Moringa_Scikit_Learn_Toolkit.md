# Getting Started with Scikit-learn for Machine Learning — A Beginner's Toolkit
### Moringa School Capstone | Purity Wanjiku Macharia

---

## 1. Title & Objective

**Technology chosen:** Scikit-learn (`scikit-learn`) — a Python machine learning library

**Why I chose it:**
I have a degree in Applied Statistics with Programming and wanted to bridge my statistical knowledge into practical, industry-standard machine learning. Scikit-learn is the most widely used ML library in the world — it appears in job descriptions for every data and ML role I have applied to. Learning it was not optional; it was necessary.

**End goal:**
Build a working end-to-end regression pipeline that predicts house prices — covering data preprocessing, feature engineering, model training, cross-validation, and submission — using real Kaggle competition data.

**Kaggle Notebook (live, runnable):**
https://www.kaggle.com/code/puritymacharia/notebook5fa5ce5d50

**GitHub Repository:**
https://github.com/pmash634/house-price-prediction

---

## 2. Quick Summary of the Technology

**What is Scikit-learn?**
Scikit-learn is an open-source Python library that provides simple, consistent tools for machine learning. It is built on top of NumPy, SciPy, and Matplotlib, and provides implementations of most classical ML algorithms out of the box.

**Where is it used?**
- Industry: fraud detection, price prediction, customer churn modelling, recommendation systems
- Research: as a baseline before deep learning is considered
- Competitions: Kaggle, DrivenData, and similar platforms

**One real-world example:**
Spotify uses regression models (similar to what we build here) to predict listener engagement and recommend songs. The same sklearn pipeline structure applies.

**Key scikit-learn concepts used in this project:**

| Concept | sklearn Class / Function |
|---|---|
| Splitting data | `train_test_split` |
| Random Forest model | `RandomForestRegressor` |
| Cross-validation | `cross_val_score` |
| Error metric | `mean_squared_error` |
| Ordinal encoding | `OrdinalEncoder` |

---

## 3. System Requirements

| Requirement | Details |
|---|---|
| Operating System | Windows 10/11, macOS 12+, or Ubuntu 20.04+ |
| Python | Version 3.8 or higher |
| Editor | VS Code or Jupyter Notebook (recommended) |
| Package manager | pip (comes with Python) |
| Internet | Required for downloading packages and Kaggle data |

**Check your Python version:**
```bash
python --version
```

---

## 4. Installation & Setup Instructions

### Step 1 — Install Python
Download from https://www.python.org/downloads/ and install.
Make sure to tick **"Add Python to PATH"** during installation on Windows.

### Step 2 — Install required packages
Open your terminal (Command Prompt on Windows, Terminal on Mac/Linux) and run:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn xgboost jupyter
```

Expected output (last few lines):
```
Successfully installed scikit-learn-1.4.0 pandas-2.1.0 numpy-1.26.0 ...
```

### Step 3 — Verify installation
```python
import sklearn
print(sklearn.__version__)  # Should print 1.4.0 or similar
```

### Step 4 — Download the dataset
1. Go to https://www.kaggle.com/c/house-prices-advanced-regression-techniques
2. Click **Join Competition** → **Data** tab → **Download All**
3. Unzip the folder — you will see `train.csv` and `test.csv`

### Step 5 — Launch Jupyter Notebook
```bash
jupyter notebook
```
This opens a browser window. Create a new Python 3 notebook and you are ready.

---

## 5. Minimal Working Example

**What this example does:**
Loads the house prices dataset, trains a Random Forest model, and prints the cross-validation RMSE — the core loop of any ML project.

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("train.csv")

# ── 2. Select simple numeric features (no preprocessing needed) ───────────────
features = ["GrLivArea", "BedroomAbvGr", "FullBath", "YearBuilt", "OverallQual"]
X = df[features].fillna(0)          # fill any missing values with 0

# ── 3. Target — log-transform to reduce skew ──────────────────────────────────
y = np.log1p(df["SalePrice"])       # log1p pairs with expm1 for reversal

# ── 4. Train a Random Forest model ───────────────────────────────────────────
model = RandomForestRegressor(n_estimators=100, random_state=42)

# ── 5. Evaluate with 5-fold cross-validation ─────────────────────────────────
scores = cross_val_score(model, X, y,
                         scoring="neg_root_mean_squared_error",
                         cv=5)
print(f"CV RMSE: {-scores.mean():.4f} ± {scores.std():.4f}")
```

**Expected output:**
```
CV RMSE: 0.1923 ± 0.0078
```

> Note: This simple version scores ~0.19. The full pipeline in the GitHub repo achieves **0.1454** through feature engineering — showing the value of the extra steps.

---

## 6. AI Prompt Journal

The following prompts were used via **ai.moringaschool.com** to learn scikit-learn and build this project.

---

**Prompt 1**
> *"I have a degree in Applied Statistics. I want to learn scikit-learn from scratch. What are the core concepts I need to understand and in what order should I learn them?"*

**AI Response Summary:**
The AI outlined a learning path: (1) understand the estimator API — fit/predict/score, (2) learn preprocessing tools, (3) learn model evaluation with cross-validation, (4) try pipelines. It explained that sklearn's consistent API means all models work the same way — once you know RandomForest, you know XGBoost's interface too.

**Evaluation:** Very helpful as a starting map. Gave me confidence that my statistics background was already covering the theory — I just needed the code patterns.

---

**Prompt 2**
> *"Explain cross_val_score to me like I am a statistics student who understands the concept of validation but has never used sklearn. Show me a code example with the house price dataset."*

**AI Response Summary:**
The AI explained that `cross_val_score` automates the process of splitting data into k folds, training on k-1 folds and testing on the remaining fold, repeating k times. It showed a code snippet using `scoring="neg_root_mean_squared_error"` and explained why sklearn uses negative values (it always maximises scores, so RMSE is negated).

**Evaluation:** Excellent. The explanation of *why* the score is negative was something the documentation does not explain clearly. This saved me significant confusion.

---

**Prompt 3**
> *"What is the difference between np.log and np.log1p for transforming a target variable? Which should I use and why?"*

**AI Response Summary:**
The AI explained that `np.log` fails if any value is 0 (returns -inf), while `np.log1p` computes log(1+x) which is safe for zero values. It stressed that the pairing must be consistent: `log` ↔ `exp`, and `log1p` ↔ `expm1`. It warned that mixing them introduces systematic errors in every prediction.

**Evaluation:** Critical. This directly fixed Error #1 in my original notebook where I had trained with `np.log` but reversed with `np.expm1` — a mismatch that was silently corrupting all predictions.

---

**Prompt 4**
> *"I want to engineer a neighbourhood median price feature for the house prices dataset. How do I do this without causing data leakage from the test set?"*

**AI Response Summary:**
The AI explained that computing the median from the full dataset (train + test combined) leaks target information. The correct approach is to compute the median only from the training set, then use `.map()` to apply it to both train and test. It provided a code example using `groupby` and `map`.

**Evaluation:** Very helpful. My original code was computing neighbourhood medians correctly from training data but then failing to include the feature in the model at all — a bug the AI helped me identify.

---

**Prompt 5**
> *"My XGBoost model is giving a FutureWarning about inplace fillna on a copy of a DataFrame in pandas. What does this mean and how do I fix it?"*

**AI Response Summary:**
The AI explained that pandas is deprecating `inplace=True` on chained assignments in pandas 3.0. It recommended replacing `df[col].fillna(value, inplace=True)` with `df[col] = df[col].fillna(value)` — direct reassignment that is unambiguous and future-proof.

**Evaluation:** Immediately actionable. Fixed the warning in one line per column.

---

## 7. Common Issues & Fixes

| Error | Cause | Fix |
|---|---|---|
| `FileNotFoundError: train.csv not found` | Dataset not attached to Kaggle notebook | In Kaggle: click ··· → Add data → search "house-prices-advanced-regression-techniques" |
| `FutureWarning: inplace fillna` | pandas 3.0 deprecation of chained inplace assignment | Replace `df[col].fillna(v, inplace=True)` with `df[col] = df[col].fillna(v)` |
| Predictions are negative or unrealistically small | Used `np.log` to train but `np.expm1` to reverse | Use matching pairs: `log` + `exp`, OR `log1p` + `expm1` |
| Test set predictions look wrong despite good CV score | Feature built differently on train vs test | Extract feature engineering into a single function applied to both datasets |
| `KeyError: 'TA'` on quality mapping | Typo in mapping dictionary (`'Ta'` instead of `'TA'`) | Define a single `QUAL_MAP` dictionary used everywhere |
| `ModuleNotFoundError: No module named 'xgboost'` | XGBoost not installed | Run `pip install xgboost` |
| `ValueError: Input contains NaN` | Null values not handled before model training | Add `.fillna()` step before fitting the model |

**Helpful resources for errors:**
- Scikit-learn common pitfalls: https://scikit-learn.org/stable/common_pitfalls.html
- Pandas FutureWarning discussion: https://stackoverflow.com/questions/71296731
- XGBoost install guide: https://xgboost.readthedocs.io/en/stable/install.html

---

## 8. References

**Official Documentation**
- Scikit-learn docs: https://scikit-learn.org/stable/
- Scikit-learn user guide (supervised learning): https://scikit-learn.org/stable/supervised_learning.html
- XGBoost Python API: https://xgboost.readthedocs.io/en/stable/python/python_api.html
- Pandas docs: https://pandas.pydata.org/docs/

**Kaggle Resources**
- House Prices competition: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
- Comprehensive data exploration notebook: https://www.kaggle.com/code/pmarcelino/comprehensive-data-exploration-with-python

**Video Tutorials**
- Scikit-learn crash course (Codebasics): https://www.youtube.com/watch?v=0B5eIE_1vpU
- XGBoost explained: https://www.youtube.com/watch?v=OtD8wVaFm6E

**My Submission**
- Live Kaggle notebook (V4): https://www.kaggle.com/code/puritymacharia/notebook5fa5ce5d50
- GitHub repo: https://github.com/pmash634/house-price-prediction
- Kaggle public score: **0.14768**
- 5-fold CV RMSE: **0.1454 ± 0.0045**

---

*Submitted by Purity Wanjiku Macharia | puritymacharia111@gmail.com | github.com/pmash634*
