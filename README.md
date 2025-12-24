# Customer Churn Prediction Using Machine Learning

##  Project Overview
Customer churn prediction is a critical problem for businesses aiming to improve customer retention and maximize profitability. This project focuses on building and evaluating machine learning models to predict whether a customer is likely to leave a banking institution based on historical customer data.

Multiple classification algorithms are implemented and compared, including Logistic Regression, Random Forest, and XGBoost. The results demonstrate that ensemble-based models, particularly XGBoost, significantly outperform traditional approaches in identifying churn-prone customers.

---

##  Problem Statement
To design and implement a machine learning model that accurately predicts customer churn using demographic, financial, and behavioral attributes.

---

##  Objectives
- Explore and preprocess customer churn data  
- Perform feature encoding and normalization  
- Implement multiple machine learning classification models  
- Evaluate and compare model performance  
- Identify key factors influencing customer churn  

---

##  Dataset Description
The project uses the **Churn_Modelling.csv** dataset containing customer information from a banking institution.

### Dataset Attributes
| Feature | Description |
|------|------------|
| CreditScore | Customer credit score |
| Geography | Customer’s country |
| Gender | Male / Female |
| Age | Age of customer |
| Tenure | Number of years with the bank |
| Balance | Account balance |
| NumOfProducts | Number of products used |
| HasCrCard | Credit card ownership |
| IsActiveMember | Activity status |
| EstimatedSalary | Annual salary |
| Exited | Target variable (1 = Churned, 0 = Retained) |

---

##  Methodology

### 1. Data Exploration
- Dataset structure analyzed using `.info()`
- Missing values checked using `.isnull()`
- Duplicate records inspected

### 2. Data Preprocessing
**Encoding**
- Gender encoded using `LabelEncoder`
- Geography converted using One-Hot Encoding  

**Feature Selection**
- Relevant numerical and categorical features selected
- Target variable excluded from feature set

**Train-Test Split**
- 80% Training data  
- 20% Testing data  

**Feature Scaling**
- `StandardScaler` applied to normalize features  
- Ensures improved convergence and model performance

---

##  Machine Learning Models Implemented

### Logistic Regression
- Linear classification model
- Used as a baseline
- Simple and interpretable but limited in handling non-linear patterns

### Random Forest Classifier
- Ensemble-based model using multiple decision trees
- Captures feature interactions and non-linear relationships
- Used for feature importance analysis

### XGBoost Classifier
- Gradient boosting–based ensemble model
- Highly efficient and scalable
- Performs well on imbalanced datasets
- Best-performing model in this project

---

##  Model Evaluation Metrics
Each model was evaluated using:
- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1-score  

Special emphasis was placed on **churn detection (Exited = 1)** due to its business importance.

---

##  Results and Discussion

### Logistic Regression Performance
- **Accuracy:** 81.1%
- **Churn Recall:** 20%
- **Churn F1-score:** 0.29  

**Observation:**  
The model performs well for non-churn customers but fails to identify most churners, making it unsuitable for real-world churn prediction.

---

### XGBoost Performance
- **Accuracy:** 86.8%
- **Churn Recall:** 50%
- **Churn F1-score:** 0.60  

**Observation:**  
XGBoost significantly improves churn detection while maintaining strong overall accuracy, making it the most reliable model.

---

### Comparative Summary

| Metric | Logistic Regression | XGBoost |
|------|--------------------|---------|
| Accuracy | 81.1% | 86.8% |
| Churn Precision | 0.55 | 0.75 |
| Churn Recall | 0.20 | 0.50 |
| Churn F1-score | 0.29 | 0.60 |

---

##  Feature Importance Analysis
Using Random Forest, the most influential features were:
- Age  
- Balance  
- Number of Products  
- Activity Status  
- Geography  

---

##  Business Insights
- Older customers with higher balances show higher churn risk  
- Inactive customers are more likely to churn  
- Customers using fewer products have higher churn probability  
- Geographic location impacts customer retention  

---

##  Conclusion
This project highlights the effectiveness of machine learning in predicting customer churn. While Logistic Regression serves as a baseline, ensemble-based models deliver significantly better performance. Among all models evaluated, **XGBoost achieves the best balance between accuracy and churn detection**, making it suitable for real-world deployment.

---

##  Future Scope
- Hyperparameter tuning using GridSearchCV  
- Handling class imbalance using SMOTE  
- Model deployment using FastAPI or Flask  
- Integration with real-time banking systems  
- Dashboard creation for churn monitoring  

---

##  Tools and Technologies
- Python  
- Pandas, NumPy  
- Matplotlib  
- Scikit-learn  
- XGBoost  

---

##  References
- Scikit-learn Documentation  
- XGBoost Documentation  
- Kaggle – Customer Churn Dataset  
- Research papers on customer churn prediction  

---

##  Author
**Shivam Rawat**  

