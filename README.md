# 🏦 Churn User Prediction

This project predicts **customer churn** for a credit card  using **Machine Learning**.  
The aim is to identify customers who are likely to leave, so that the bank can take proactive steps to retain them.

---

## 📌 Project Overview
Customer churn is a major concern in the banking industry — acquiring new customers costs significantly more than retaining existing ones.  
By analyzing customer demographics, account details, and behavior, we can build a model that predicts the probability of churn.

This repository includes:
- Data preprocessing & cleaning
- Feature engineering
- Model training & evaluation
- Insights & visualizations

---

## 📂 Dataset Description

### **Original Features**
| Feature           | Description |
|-------------------|-------------|
| `CustomerID`      | Unique customer identifier (not used for modeling). |
| `Gender`          | Male / Female / Unknown. |
| `Age`             | Customer's age in years. |
| `Tenure`          | Number of years with the bank. |
| `Balance`         | Account balance. |
| `NumOfProducts`   | Number of products used (savings account, credit card, loans, etc.). |
| `HasCrCard`       | 1 if the customer has a credit card with the bank, else 0. |
| `IsActiveMember`  | 1 if the customer regularly uses the bank’s services, else 0. |
| `EstimatedSalary` | Estimated annual salary. |
| `Churn`           | Target variable — 1 if the customer left, 0 if retained. |

---

### **Engineered Features**
| Feature                  | Description |
|--------------------------|-------------|
| `BalancePerProduct`      | Average balance per product. |
| `AgeBand`                | Age grouped into ranges (18–29, 30–39, etc.). |
| `TenureBand`             | Tenure grouped into ranges (0–2, 3–5, 6+ years). |
| `ActivityLevel`          | Engagement score (Low/Medium/High) from `HasCrCard` & `IsActiveMember`. |
| `SalaryPerProduct`       | Estimated salary per product. |
| `BalanceToSalaryRatio`   | Ratio of balance to salary — a financial health indicator. |
| `HighValueCustomer`      | Flag for customers with balance or salary in top 25%. |
| `CustomerRiskScore`      | Composite churn risk score based on age, tenure, products, and activity. |
| `HasZeroBalance`         | 1 if balance = 0. |
| `IsSingleProduct`        | 1 if only one product is used. |
| `LogBalance`             | Log-transformed balance to reduce skew. |
| `LogSalary`              | Log-transformed salary to reduce skew. |

---

## ⚙️ Workflow
1. **Data Cleaning**  
   - Handle missing values  
   - Remove or cap outliers  
   - Convert data types  

2. **Feature Engineering**  
   - Create new behavioral, ratio, and categorical features  
   - Transform skewed features using log transformation  
   - Encode categorical variables  

3. **Modeling**  
   - Train/test split  
   - Algorithms tested: Logistic Regression, Random Forest, XGBoost, etc.  
   - Hyperparameter tuning  

4. **Evaluation**  
   - Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC  
   - Feature importance analysis  
   - Visualizations of churn patterns  

---

## 📊 Key Insights
- **Inactive members** have significantly higher churn rates.  
- Customers with **only one product** are at higher churn risk.  
- **High balance per product** customers are more loyal.  
- Age group **30–39** shows higher churn than older age bands.  

---

## 🚀 Tech Stack
- **Language:** Python 3.x  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Environment:** Jupyter Notebook / Google Colab  


