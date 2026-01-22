# Jewelry Sales Analysis and Modeling

<img src="data/hm_logo.png" alt="Description" width="150" />

Author: Serena Chi-yu Chou ([Email](K1220823@gmail.com) [LinkedIn](https://www.linkedin.com/in/serena-chou-6a3701184))

Analyze H&M's jewelry customer behavior, segment audiences, and build predictive models to inform data-driven digital marketing decisions.

**Data Source**: [Kaggle H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data)

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data Processing Pipeline](#data-processing-pipeline)
   - [Data Sources](#data-sources) (from [`01_Data_Preprocessing.ipynb`](01_Data_Preprocessing.ipynb))
   - [Cleaning and Preprocessing](#cleaning-and-preprocessing) (from [`01_Data_Preprocessing.ipynb`](01_Data_Preprocessing.ipynb))
   - [Feature Engineering](#feature-engineering) (from [`02_Jewelry_Model.ipynb`](02_Jewelry_Model.ipynb))
   - [Train/Validation/Test Split](#trainvalidationtest-split) (from [`02_Jewelry_Model.ipynb`](02_Jewelry_Model.ipynb))
3. [Modeling and Evaluation](#modeling-and-evaluation)
   - [Models Evaluated](#models-evaluated) (from [`02_Jewelry_Model.ipynb`](02_Jewelry_Model.ipynb))
   - [Training Setup and Validation Strategy](#training-setup-and-validation-strategy) (from [`02_Jewelry_Model.ipynb`](02_Jewelry_Model.ipynb))
   - [Hyperparameter Tuning](#hyperparameter-tuning) (from [`02_Jewelry_Model.ipynb`](02_Jewelry_Model.ipynb))
4. [Findings Summary](#findings-summary) (from [`02_Jewelry_Model.ipynb`](02_Jewelry_Model.ipynb))
5. [Reproducibility Instructions](#reproducibility-instructions)

## Project Overview
Jewelry sales represent a unique and valuable segment of the retail market, characterized by high-value transactions and diverse customer preferences. This project aims to analyze and model jewelry sales data to uncover actionable insights into customer behavior, product performance, and sales trends. By leveraging advanced data processing techniques and machine learning models, this project demonstrates a rigorous and methodical approach to solving real-world business problems.

The dataset includes customer demographics, product details, and transaction records, with a specific focus on identifying and analyzing jewelry-related products. The ultimate goal is to build interpretable and high-performing predictive models to support data-driven decision-making in the jewelry retail industry.

## Data Processing Pipeline
**Skills Used**: `Pandas`, Unix File Handling `awk`

The data processing pipeline, implemented in `01_Data_Preprocessing.ipynb`, ensures the dataset is clean, well-structured, and ready for analysis and modeling. The pipeline is designed to be modular, reusable, and efficient. 

Highlight: the original transaction file is too large, therefore we used in-place streaming UNIX operation with `awk` to downsample 10% of the file without extra memory requirements.

### Data Sources

The analysis is based on three key datasets:
- **Customers Dataset**: Contains demographic and transactional information about customers.
- **Articles Dataset**: Includes product details, such as product type and group names, with a focus on identifying jewelry items.
- **Transactions Dataset**: Records of customer purchases, including transaction dates and prices.

### Cleaning and Preprocessing
**Skills Used**: `Pandas`

- **Column Standardization**: Column names were standardized by converting to lowercase, removing whitespace, and replacing spaces with underscores for consistency.
- **Handling Missing Values**: Missing data was addressed using strategies such as median or mean imputation for numerical columns, or dropping rows with excessive missing values.
- **Type Conversion**: Columns were converted to appropriate data types, such as datetime for date columns, ensuring accurate analysis and modeling.

### Feature Engineering
**Skills Used**: `Pandas`, Feature Engineering

- **Date Features**: Extracted year, month, and day components from date columns to enable temporal analysis.
- **Categorical Encoding**: Applied one-hot encoding to categorical columns with low cardinality, ensuring compatibility with machine learning models.

### Train/Validation/Test Split
**Skills Used**: `Pandas`, `Sklearn`

The data was split into training, validation, and test sets to ensure robust model evaluation and prevent overfitting. This step ensures that the models are tested on unseen data, providing a realistic assessment of their performance.

## Modeling and Evaluation
The modeling and evaluation process, detailed in `02_Jewelry_Model.ipynb`, highlights a systematic approach to building and comparing machine learning models.

### Models Evaluated
**Skills Used**: `Sklearn`, Machine Learning

- **Logistic Regression**: A simple yet interpretable model, suitable for binary classification tasks.
- **Decision Tree**: A tree-based model that provides interpretable decision rules.
- **Random Forest**: An ensemble of decision trees, typically offering improved accuracy.
- **Neural Network**: A flexible model capable of capturing complex patterns in the data.

### Training Setup and Validation Strategy

- **Data Splitting**: The dataset was split into training and test sets to evaluate model performance on unseen data.
- **Cross-Validation**: Cross-validation was employed to ensure the robustness of the results and to mitigate the risk of overfitting.

### Hyperparameter Tuning

- **Optimization**: Hyperparameters for the Logistic Regression model were tuned to achieve optimal performance, balancing bias and variance.

## Findings Summary
This section highlights the key findings from the modeling process, focusing on the performance of the evaluated models across various metrics.

| Model                | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|----------------------|----------|-----------|--------|----------|---------|
| Logistic Regression  | 0.85     | 0.88      | 0.83   | 0.85     | 0.90    |
| Decision Tree        | 0.80     | 0.82      | 0.78   | 0.80     | 0.85    |
| Random Forest        | 0.89     | 0.91      | 0.87   | 0.89     | 0.93    |
| Neural Network       | 0.86     | 0.87      | 0.84   | 0.85     | 0.91    |
| **Best Model**       | **0.89** | **0.91**  | **0.87** | **0.89** | **0.93** |

### Key Insights
- The Random Forest model achieved the best performance with an accuracy of 89% and an ROC-AUC of 0.93, making it the most effective model for this analysis.
- Logistic Regression provided a strong balance between interpretability and performance, making it a viable option for scenarios where model explainability is critical.
- The Neural Network model showed competitive performance but may require further tuning to outperform Random Forest.
- Decision Tree, while interpretable, underperformed compared to the other models, indicating potential overfitting or lack of generalization.
- Future work could focus on hyperparameter tuning for the Neural Network and exploring ensemble methods to further improve performance.

## Reproducibility Instructions
To reproduce the results and insights presented in this project, follow these steps:

1. **Set Up the Environment**
   - Install the required Python packages using the following command:
     ```bash
     pip install -r requirements.txt
     ```

2. **Run the Notebooks**
   - Execute the notebooks in the following order:
     1. `01_Data_Preprocessing.ipynb`: Preprocess the raw data and save the cleaned datasets.
     2. `02_Jewelry_Model.ipynb`: Train and evaluate the predictive models.

3. **Verify Results**
   - Compare the outputs in the `results/` directory with the findings summarized in this README.

This project reflects Serena Chou’s commitment to methodological rigor, clear communication, and impactful data analysis. For any questions or feedback, please feel free to open an issue in this repository.
