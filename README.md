# Jewelry Customer Behavior Analytics & Predictive Modeling

<img src="data/hm_logo.png" alt="H&M Logo" width="150" />

**Author:** Serena Chi-Yu Chou  
📧 Email: K1220823@gmail.com  
🔗 LinkedIn: https://www.linkedin.com/in/serena-chou-6a3701184  

---

## 📊 Project Overview

This project analyzes customer purchasing behavior in the jewelry segment of a large retail dataset to support data-driven marketing and business decision-making. The objective is to identify customer patterns, understand factors influencing purchase behavior, and develop predictive models that can help guide customer targeting strategies.

Rather than focusing solely on model performance, this project emphasizes how real-world customer data can be cleaned, analyzed, and translated into actionable business insights.

**Business Problem:**  
Retailers often face challenges in identifying high-value customers and optimizing marketing targeting. This project demonstrates how customer transaction data can be leveraged to improve segmentation strategies and support more effective digital marketing decisions.

**Dataset:**  
H&M Personalized Fashion Recommendations Dataset (Kaggle)

---

## 🔄 Data Processing & Analytics Workflow

This project follows a structured data analytics pipeline designed to handle large-scale retail data and simulate real-world business analytics workflows.

### Key Steps:

### 1️⃣ Data Cleaning & Preprocessing
- Standardized column formats and handled missing values
- Converted raw transaction data into analysis-ready structures
- Used UNIX streaming (`awk`) to efficiently downsample large datasets without memory overload

**Skills Demonstrated:**
- Data cleaning with Pandas
- Handling large datasets efficiently
- Real-world data preprocessing workflows

---

### 2️⃣ Exploratory Data Analysis (EDA)
- Analyzed customer demographics and purchase patterns
- Identified key behavioral indicators linked to jewelry purchases
- Examined temporal trends and product category relationships

**Skills Demonstrated:**
- Data exploration
- Behavioral pattern analysis
- Business-oriented data interpretation

---

### 3️⃣ Feature Engineering
- Created temporal purchase indicators
- Encoded categorical variables for modeling
- Constructed customer behavioral features

---

### 4️⃣ Predictive Modeling
Several models were evaluated to identify customers likely to purchase jewelry products:

- Logistic Regression
- Decision Tree
- Random Forest
- Neural Network

The goal was not only to compare accuracy, but to assess model interpretability and practical business applicability.

---

## 📈 Key Business Insights

This analysis revealed several insights relevant to real-world marketing and customer strategy:

- Customer engagement frequency strongly correlates with jewelry purchasing behavior.
- High-value customers exhibit distinct behavioral patterns that can be used to guide targeted marketing campaigns.
- Certain demographic and behavioral features serve as strong predictors of jewelry purchase likelihood.
- Customer segmentation analysis indicates opportunities for personalized outreach strategies to improve conversion rates.

These findings demonstrate how customer data analytics can support more precise marketing targeting and improved business decision-making.

---

## 💼 Business Impact & Practical Applications

This project illustrates how data analytics can support real-world business decisions, including:

- Identifying high-value customer segments
- Improving marketing targeting efficiency
- Supporting customer retention strategies
- Enhancing data-driven decision-making in retail environments

The workflow reflects practical data analyst responsibilities such as data cleaning, behavioral analysis, model interpretation, and translating insights into business recommendations.

---

## 🛠️ Tools & Skills Demonstrated

- Python (Pandas, NumPy, Scikit-learn)
- Data cleaning and preprocessing
- Exploratory data analysis
- Feature engineering
- Predictive modeling
- Business insight interpretation
- Handling large datasets

---

## 📂 Project Structure

- `01_Data_Preprocessing.ipynb` – Data cleaning and preprocessing pipeline
- `02_Jewelry_Model.ipynb` – Modeling, evaluation, and analysis
- `results/` – Model outputs and evaluation results

---

## 🔁 Reproducibility

To reproduce this project:

```bash
pip install -r requirements.txt
