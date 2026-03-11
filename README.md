# Jewelry Sales Analysis and Modeling

<img src="data/hm_logo.png" alt="Description" width="150" />

Author: Serena Chi-Yu Chou ([Email](mailto:K1220823@gmail.com) | [LinkedIn](https://www.linkedin.com/in/serena-chou-6a3701184))

Analyze H&M's jewelry customer behavior, segment audiences, and build predictive models to inform data-driven digital marketing decisions.

**Data Source**: [Kaggle H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data)

---

## Executive Project Presentation

This slide deck provides a business-focused overview of the analysis, highlighting key customer insights, modeling results, and strategic recommendations.

👉 [View Business Insights & Strategy Deck](https://drive.google.com/file/d/1Gg1WW2XOQD5-wljZgJaezi8YqN10Y6NI/view?usp=drive_link)

---

# Table of Contents

1. [Project Overview](#project-overview)  
2. [Data Processing Pipeline](#data-processing-pipeline)  
   - [Data Sources](#data-sources) (from [`01_Data_Preprocessing.ipynb`](01_Data_Preprocessing.ipynb))  
   - [Cleaning and Preprocessing](#cleaning-and-preprocessing) (from [`01_Data_Preprocessing.ipynb`](01_Data_Preprocessing.ipynb))  
   - [Feature Engineering](#feature-engineering) (from [`02_Jewelry_Model_Finalised.ipynb`](02_Jewelry_Model_Finalised.ipynb))  
   - [Train / Validation / Test Split](#train-validation-test-split) (from [`02_Jewelry_Model_Finalised.ipynb`](02_Jewelry_Model_Finalised.ipynb))  
3. [Modeling and Evaluation](#modeling-and-evaluation)  
   - [Models Evaluated](#models-evaluated) (from [`02_Jewelry_Model_Finalised.ipynb`](02_Jewelry_Model_Finalised.ipynb))  
   - [Threshold Selection Strategy](#threshold-selection-strategy)  
   - [Neural Network Hyperparameter Tuning](#neural-network-hyperparameter-tuning)  
   - [Final Model Evaluation](#final-model-evaluation)  
4. [Findings Summary](#findings-summary)  
5. [Reproducibility Instructions](#reproducibility-instructions)

---

# Project Overview

Jewelry products represent a valuable segment within the fashion retail market. Understanding which customers are most likely to purchase jewelry allows retailers to design targeted marketing campaigns and improve product positioning.

This project builds a **machine learning pipeline to predict jewelry purchase propensity** using the H&M transaction dataset. Transaction-level purchase records are transformed into customer-level behavioral features and used to train multiple classification models.

The goals of the project are:

- Identify behavioral signals associated with jewelry purchasing
- Compare multiple machine learning models
- Implement a structured evaluation framework
- Demonstrate a reproducible machine learning workflow

---

# Data Processing Pipeline

**Skills Used:** `Pandas`, Unix File Handling (`awk`)

The raw H&M transaction dataset is extremely large. To enable efficient analysis, a **streaming Unix pipeline using `awk`** was used to downsample approximately 10% of the dataset without loading the full file into memory.

The preprocessing pipeline is implemented in: 01_Data_Preprocessing.ipynb


---

## Data Sources

The analysis integrates three primary datasets:

### Customers Dataset

Contains customer demographic information and unique identifiers.

### Articles Dataset

Includes product metadata such as:

- product type  
- product group name  
- article description  

These attributes are used to identify jewelry items.

### Transactions Dataset

Contains transaction-level purchase records including:

- `customer_id`
- `article_id`
- `price`
- `transaction_date`

---

## Cleaning and Preprocessing

Key preprocessing steps include:

### Column Standardization

Column names were standardized by:

- converting to lowercase  
- removing whitespace  
- replacing spaces with underscores  

### Handling Missing Values

Missing values were handled using:

- mean or median imputation for numeric columns
- removal of rows with excessive missing data

### Type Conversion

Date columns were converted to `datetime` format to support temporal feature extraction.

---

## Feature Engineering

Transaction-level data was aggregated into **customer-level behavioral features**.

Feature categories include:

- purchase frequency
- total spending
- average purchase value
- product diversity

Example aggregation logic:

```python
customer_features = transactions.groupby("customer_id").agg({
    "price": ["count", "sum", "mean"],
    "article_id": "nunique"
})
These features form the model input matrix used for training machine learning classifiers.

## Train / Validation / Test Split

The dataset is divided using a **stratified split** to ensure class balance across all subsets.

| Dataset | Purpose |
|--------|--------|
| Train (70%) | Model training |
| Validation (15%) | Threshold selection and model tuning |
| Test (15%) | Final model evaluation |

Stratification ensures that the proportion of positive and negative classes remains consistent across splits.

This design prevents **data leakage** and allows unbiased evaluation on unseen data.

---

# Modeling and Evaluation

The modeling workflow is implemented in:

`02_Jewelry_Model_Finalised.ipynb`

Multiple machine learning models are trained and compared using a consistent evaluation framework.

---

## Models Evaluated

Four classification models were evaluated.

### Logistic Regression
A linear baseline classifier that provides strong interpretability and serves as a benchmark model.

### Decision Tree
A tree-based classifier capable of generating interpretable decision rules based on feature splits.

### Random Forest
An ensemble learning method that combines multiple decision trees to improve predictive performance and reduce overfitting.

### Neural Network (MLPClassifier)
A multi-layer perceptron capable of capturing nonlinear relationships in customer behavior patterns.

---

## Threshold Selection Strategy

Most classifiers output predicted probabilities rather than class labels.

To determine the optimal classification boundary, a **threshold sweep** was performed on the validation set for the following models:

- Logistic Regression  
- Decision Tree  
- Random Forest  

Example thresholds evaluated: 0.1, 0.2, 0.3 ... 0.9


Performance metrics such as **precision, recall, and F1-score** were evaluated across thresholds to determine the most suitable operating point.

The selected threshold was then applied when evaluating the model on the test dataset.

---

## Neural Network Hyperparameter Tuning

Unlike the other models, the neural network was tuned using **RandomizedSearchCV**.

The workflow was:

1. Perform hyperparameter search using the validation fold  
2. Identify the best hyperparameter configuration  
3. Retrain the neural network using **Train + Validation data**  
4. Evaluate the final model on the **held-out test set**

This approach ensures that the test dataset remains completely unseen during model selection.

---

## Final Model Evaluation

Model performance is evaluated using multiple metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC

Final model comparison is performed **only on the test set** to ensure unbiased out-of-sample performance estimation.

---

# Findings Summary

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|------|------|------|------|------|------|
| Logistic Regression | 0.688458 | 0.021912 | 0.540866 | 0.042118 | 0.656260 |
| Neural Network (best params) | 0.745550 | 0.023273 | 0.466056 | 0.044333 | 0.654290 |
| Decision Tree | 0.685353 | 0.021345 | 0.531711 | 0.041042 | 0.640591 |
| Random Forest | 0.787717 | 0.023890 | 0.395489 | 0.045058 | 0.635691 |
| **Best Model** | **Random Forest** | **0.023890** | **0.395489** | **0.045058** | **0.635691** |


Actual results can be reproduced by executing the modeling notebook.

General observations include:

- Ensemble models typically outperform single decision trees.
- Logistic regression provides strong baseline interpretability.
- Neural networks capture nonlinear behavioral signals but require careful tuning.

---

# Reproducibility Instructions

To reproduce the analysis:

### 1. Install Dependencies
pip install -r requirements.txt

### 2. Run the Notebooks

Execute the notebooks in the following order:

01_Data_Preprocessing.ipynb
02_Jewelry_Model_Finalised.ipynb



The preprocessing notebook prepares the modeling dataset.

The modeling notebook performs:

- train / validation / test split  
- model training  
- threshold sweep  
- neural network hyperparameter tuning  
- final test evaluation  

---

### 3. Verify Results

Outputs generated in the notebook should match the evaluation metrics summarized in this repository.

---

# Project Significance

This project demonstrates a complete applied machine learning workflow including:

- large-scale data preprocessing
- feature engineering from transactional retail data
- structured model comparison
- hyperparameter tuning
- unbiased out-of-sample evaluation

The methodology reflects best practices used in real-world retail analytics and predictive modeling.

This project reflects Serena Chou’s commitment to methodological rigor, clear communication, and impactful data analysis. For any questions or feedback, please feel free to open an issue in this repository.
