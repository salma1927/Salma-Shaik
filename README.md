# Salma-Shaik
# Predicting Heart Disease Risk with Machine Learning 
A Comprehensive Analysis of Heart Disease Prediction Using ML Models

Heart Disease Prediction

Project Overview
Cardiovascular disease (CVD) is one of the leading causes of mortality worldwide. This project explores the use of Machine Learning (ML) models to predict the risk of heart disease based on various health and lifestyle factors. By leveraging Support Vector Classifier, Gradient Boosting, and Extra Trees Classifier, we aim to build an accurate predictive model that helps in early detection and risk assessment.  

🔹 Goal: Predict the likelihood of an individual having heart disease.  
🔹 Data Source:Behavioral Risk Factor Surveillance System (BRFSS) 2020 (via Kaggle).  
🔹 Models Used: Support Vector Classifier (SVC), Gradient Boosting, Extra Trees Classifier.  
🔹 Tech Stack: Python, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn.  

---

## Dataset Description
The dataset was obtained from Kaggle, originally sourced from the CDC’s Behavioral Risk Factor Surveillance System (BRFSS) 2020. It includes 20 key health indicators relevant to heart disease risk.  

### Key Features:
- Binary Features:Smoking, AlcoholDrinking, Stroke, DiffWalking, PhysicalActivity, Asthma, KidneyDisease, SkinCancer.  
- Categorical Features: Sex, Race (6 categories), Diabetic Status, AgeCategory, GenHealth (self-reported health status).  
- Continuous Features: BMI, PhysicalHealth, MentalHealth, SleepTime.  

---

## Research Questions & Hypotheses
 Can a machine learning model accurately predict heart disease risk?
 What are the most significant features influencing heart disease predictions? 
 How can we handle class imbalance in medical datasets?

---

## Machine Learning Approach  

###  1️.Data Preprocessing & Feature Engineering
✔ Handled missing values & categorical encoding  
✔ Addressed class imbalance using RandomUnderSampler & NearMiss  
✔ Performed feature scaling for optimal model performance  

###  2. ML Models Used & Rationale
| Model | Reason for Selection | Pros | Cons |
|--------|--------------------|------|------|
| Support Vector Classifier (SVC)| Works well in high-dimensional spaces | High accuracy | Computationally expensive |
| Gradient Boosting Classifier| Strong predictive performance | Handles complex patterns well | Prone to overfitting |
| Extra Trees Classifier | Handles feature importance well | Resistant to overfitting | Less interpretable |

 Cross-validation: Used K-Fold Cross Validation to validate model robustness.

---

##  Implementation Steps
### 1️⃣ Import Libraries
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from imblearn.under_sampling import RandomUnderSampler, NearMiss
```

### 2️⃣ Load Dataset
```python
df = pd.read_csv("heart_disease_data.csv")
df.head()
```

### 3️⃣ Data Preprocessing
- Categorical Encoding
- Feature Scaling
- Handling Class Imbalance

### 4️⃣ Model Training & Evaluation
```python
# Splitting dataset
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Model Prediction
y_pred = model.predict(X_test)

# Evaluate Performance
from sklearn.metrics import accuracy_score, classification_report
print(classification_report(y_test, y_pred))
```

---

##  Results & Insights
✔ Gradient Boosting performed best with highest accuracy. 
✔ Feature Importance Analysis: Age, BMI, and General Health were top predictors.  
✔ Undersampling techniques improved model balance and precision.

---

##  Challenges & Lessons Learned 
🔹 Class imbalance was a major issue – Addressed with RandomUnderSampler.  
🔹 Feature selection required domain knowledge – Consulting medical research helped refine key variables.  
🔹 Hyperparameter tuning improved performance – GridSearchCV was used for model optimization.  

---

##  Future Enhancements
🔸 Experiment with Deep Learning Models (LSTMs, CNNs)
🔸 Use SHAP values for better feature interpretability 
🔸 Deploy model using Flask or Streamlit for a web-based application 

---

##  How to Use This Project? 
1️⃣ Clone the repository  
```bash
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction
```
2️⃣ Install required dependencies  
```bash
pip install -r requirements.txt
```
3️⃣ Run the ML model  
```bash
python heart_disease_model.py
```

---

##  References 
- CDC Behavioral Risk Factor Surveillance System: [BRFSS 2020](https://www.cdc.gov/brfss/)  
- Kaggle Dataset: [Heart Disease Prediction](https://www.kaggle.com/)  
- Machine Learning Libraries: Scikit-learn, Pandas, NumPy  

---
