# Jewelry Purchase Propensity Model

<img src="data/hm_logo.png" alt="H&M Logo" width="150" />

**Author:** Serena Chi-yu Chou &nbsp;|&nbsp; [Email](mailto:K1220823@gmail.com) &nbsp;|&nbsp; [LinkedIn](https://www.linkedin.com/in/serena-chou-6a3701184)

> Predicting which H&M customers are likely to purchase jewelry in their next transaction — using transaction-level supervised learning on 3.1M+ records, with careful handling of class imbalance and data leakage.

**Data Source:** [Kaggle — H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data)

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Dataset Description](#2-dataset-description)
3. [Problem Statement](#3-problem-statement)
4. [Methodology & Modeling Steps](#4-methodology--modeling-steps)
5. [Evaluation Metrics & Results](#5-evaluation-metrics--results)
6. [Key Business Insights](#6-key-business-insights)
7. [Tools & Libraries](#7-tools--libraries)
8. [How to Run](#8-how-to-run)

---

## 1. Project Overview

This project builds a **binary classification model** to predict jewelry purchase propensity at the transaction level: given a customer's current transaction and full purchase history, will their **next** transaction be a jewelry purchase?

The model is designed to support data-driven marketing decisions — enabling the business to identify high-propensity customers for targeted jewelry campaigns with measurably higher conversion rates than random outreach.

**Key challenges addressed:**
- Severe class imbalance (1.3% positive rate, ~78:1 ratio)
- Preventing data leakage through strict temporal feature construction
- Selecting evaluation metrics appropriate for imbalanced, business-facing use cases (PR-AUC, Top-K Lift)

---

## 2. Dataset Description

Three linked datasets from the H&M Kaggle competition, spanning **September 2018 – September 2020**:

| Dataset | Rows | Columns | Key Fields |
|---|---|---|---|
| Customers | 1,371,980 | 8 | `customer_id`, `age`, `club_member_status`, `fashion_news_frequency` |
| Articles | 105,542 | 26 | `article_id`, `product_type_name`, `product_group_name`, `color_group_name` |
| Transactions | 3,178,778 | 6 | `t_dat`, `customer_id`, `article_id`, `price`, `primary_sales_channel` |

**Jewelry product breakdown** (2,409 items, 2.28% of catalog):

| Type | Count |
|---|---|
| Earring | 1,159 |
| Necklace | 581 |
| Ring | 240 |
| Hair string | 238 |
| Bracelet | 180 |

The **transaction-level ML dataset** contains **2,357,449 observations** across 821,329 unique customers (customers with at least 1 transaction).

---

## 3. Problem Statement

### Class Imbalance
Jewelry purchases account for only **1.3% of all transactions** (29,850 positive vs. 2,327,599 negative), a ~78:1 class ratio. Standard accuracy is misleading here — a model predicting "never jewelry" achieves ~98.7% accuracy while being entirely useless for targeting.

**Mitigation strategies applied:**
- `class_weight='balanced'` in all models (inverse-frequency weighting)
- Stratified train/val/test splits to preserve class distribution
- Threshold-independent metrics: ROC-AUC and PR-AUC
- Business-oriented Top-K Lift evaluation (1% and 2% targeting thresholds)

### Data Leakage Prevention
The target is defined as whether the customer's **next** transaction is jewelry — so each observation must only contain information from **prior** transactions.

**Safeguards implemented:**
- All features are aggregated strictly from historical transactions (before the current row)
- The **last transaction per customer is dropped** (no future label available)
- All preprocessing (imputation, encoding) is **fit on training data only**, then applied to validation and test sets
- Recency is measured as days since the *previous* transaction, not the current one

---

## 4. Methodology & Modeling Steps

### Step 1 — Data Preprocessing (`01_Data_Preprocessing.ipynb`)
- Standardized column names and data types (`datetime`, `float32`, `Int8`)
- Handled missing values: median imputation for numeric, most-frequent for categorical
- Downsampled the raw transaction file (3.1M+ rows) using in-place Unix `awk` streaming to avoid memory bottlenecks

### Step 2 — Feature Engineering (`02_Jewelry_Model_Finalised.ipynb`)
Features are built at three levels and joined at the transaction level:

**Customer-level (14 features):**
- Purchase behavior: `purchase_frequency`, `total_spending`, `average_order_value`, `recency_days`
- Jewelry history: `jewelry_purchase_count`, `jewelry_total_spending`, `jewelry_avg_price`
- Demographics: `age`, `age_group` (7 bins), `club_member_status`, `fashion_news_frequency`
- Channel preference: `primary_sales_channel`

**Product-level (34 features):**
- Category metadata: `product_group`, `department`, `color_group`
- Popularity: `purchase_count`, `total_revenue`, `unique_customers_reached`

**Interaction-level:**
- Customer × category recency (days since last purchase per category)
- Product diversity metrics: avg 3.7 unique categories, 2.2 product groups, 8.1 unique products per customer

### Step 3 — ML Dataset Design
- Unit of observation: **transaction-level** (each row = one transaction event)
- Target variable: `next_is_jewelry` (binary, 1 if the customer's next transaction is jewelry)
- Final dataset: 2,357,449 rows × engineered feature set

### Step 4 — Train/Validation/Test Split
Stratified split preserving the 1.3% positive rate:

| Split | Rows |
|---|---|
| Train (70%) | 1,650,214 |
| Validation (15%) | 353,617 |
| Test (15%) | 353,618 |

Models were trained on train set, tuned on validation, then retrained on train+validation before final test evaluation.

### Step 5 — Models

**Logistic Regression (baseline):**
- `solver='lbfgs'`, `max_iter=1000`, `class_weight='balanced'`
- Establishes a linear probability baseline; interpretable log-odds coefficients

**Decision Tree:**
- `max_depth=4`, `min_samples_leaf=50`, `class_weight='balanced'`
- 31 nodes / 16 leaves; fully interpretable business rules
- Top leaf: customers with prior jewelry spending + 7+ unique products → **3.46% jewelry rate** (2.7× base rate)

---

## 5. Evaluation Metrics & Results

Given severe class imbalance, evaluation focuses on **PR-AUC** and **Top-K Lift** rather than accuracy.

### Model Comparison (Test Set)

| Metric | Logistic Regression | Decision Tree |
|---|---|---|
| Accuracy | 0.689 | 0.685 |
| Precision | 0.022 | 0.021 |
| Recall | 0.541 | 0.532 |
| F1-Score | 0.042 | 0.041 |
| ROC-AUC | **0.656** | 0.641 |
| PR-AUC | **0.030** | 0.026 |

> Accuracy is near 69% for both models — this reflects the class distribution, not meaningful signal. PR-AUC and Lift are the relevant metrics here.

### Top-K Lift (Test Set, targeting top 2% of predictions)

| Model | Precision@2% | Recall@2% | Lift |
|---|---|---|---|
| Logistic Regression | 6.2% | 9.8% | **4.9×** |
| Decision Tree | 6.5% | 10.3% | **5.1×** |

**Interpretation:** By targeting the top 2% of customers ranked by model score, the business captures **5× more jewelry buyers** than random outreach — making every marketing dollar roughly 5 times more efficient.

---

## 6. Key Business Insights

1. **Use a low prediction threshold (1–2%) for campaign targeting** rather than the default 0.5 cutoff. This delivers 6–7% precision with 5× lift — a practical trade-off between reach and targeting efficiency.

2. **Prior jewelry spend is the strongest signal.** Customers with any jewelry purchase history and 7+ unique products purchased have a jewelry propensity rate of 3.46% — nearly 3× the base rate.

3. **Channel 2 customers show higher jewelry propensity.** Channel-specific marketing (e.g., online vs. in-store) can further improve targeting precision.

4. **Decision Tree is preferred for business deployment** due to full interpretability. Decision rules can be validated and explained to non-technical stakeholders (e.g., "target customers aged <37 with prior jewelry spending and diverse product history").

5. **Age segmentation matters.** Customers aged ≤33–37 exhibit different purchase propensity patterns and should receive tailored messaging.

---

## 7. Tools & Libraries

| Category | Tools |
|---|---|
| Data Manipulation | `pandas`, `numpy` |
| Machine Learning | `scikit-learn` 1.7.0 (LogisticRegression, DecisionTreeClassifier, Pipeline, ColumnTransformer, SimpleImputer, OneHotEncoder) |
| Evaluation | `sklearn.metrics`: `roc_auc_score`, `average_precision_score`, `precision_recall_curve`, `confusion_matrix` |
| Visualization | `matplotlib`, `seaborn` |
| Environment | Google Colab, Python 3.12 |
| Data Engineering | Unix `awk` (in-place streaming downsampling) |

---

## 8. How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Data Setup
Download the raw data from [Kaggle](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data) and place the files in the `data/` directory:
```
data/
├── customers.csv
├── articles.csv
└── transactions_train.csv
```

### Run Notebooks in Order
```
1. 01_Data_Preprocessing.ipynb   — Clean raw data, downsample transactions, save processed files
2. 02_Jewelry_Model_Finalised.ipynb  — Feature engineering, model training, evaluation
```

> Both notebooks are designed to run on **Google Colab** with data stored on Google Drive. Update the `DRIVE_PATH` variable at the top of each notebook to match your Drive directory.

### Expected Outputs
- Processed feature datasets saved to `data/processed/`
- Model evaluation metrics printed inline (accuracy, PR-AUC, ROC-AUC, Top-K Lift)
- ROC and Precision-Recall curve plots

---

*This project reflects a commitment to methodological rigor — from leakage-free feature design to business-grounded evaluation. For questions or feedback, feel free to open an issue or reach out via [LinkedIn](https://www.linkedin.com/in/serena-chou-6a3701184).*
