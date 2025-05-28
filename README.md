# Titanic Survival Prediction

## **Project Overview**

This project tackles the classic Titanic dataset from Kaggle’s "Titanic: Machine Learning from Disaster" competition. The objective is to predict which passengers survived the Titanic shipwreck using features such as age, gender, class, fare, and more. This serves as a foundational machine learning classification task, ideal for demonstrating end-to-end data science skills.

---

## **1. Problem Statement**

Predict whether a given passenger survived the Titanic disaster (binary classification: 1 = survived, 0 = did not survive) using available passenger and ticket data.

---

## **2. Dataset Description**

- **Train Set:** Contains features and the target variable (`Survived`).
- **Test Set:** Contains features only; the model predicts `Survived` for these passengers.
- **Key Features:**
  - `Pclass`: Passenger class (1st, 2nd, 3rd)
  - `Sex`: Gender
  - `Age`: Age in years
  - `SibSp`: # of siblings/spouses aboard
  - `Parch`: # of parents/children aboard
  - `Fare`: Ticket fare
  - `Embarked`: Port of embarkation (C, Q, S)
  - `Cabin`, `Ticket`, `Name`: Additional info, some with many missing values

---

## **3. Data Preprocessing**

- **Missing Value Imputation:**
  - Used `IterativeImputer` for `Age` and `Fare` (numerical features with missing values).
  - For `Embarked`, filled missing values with the mode.
  - Dropped the `Cabin` feature due to excessive missingness.
- **Feature Engineering:**
  - **Label Encoding:** Converted `Sex` to numeric (`Sex_encoded`).
  - **One-Hot Encoding:** Applied to `Embarked` and `Pclass`.
  - **Log Transformation:** Applied `log1p` to `SibSp`, `Parch`, and `Fare` to reduce skewness.
  - **Feature Scaling:** Used `StandardScaler` for numerical features.
- **Outlier Handling:** Detected outliers in `Fare` and handled them as needed.
- **Column Selection:** Selected relevant features for modeling, dropping identifiers and high-missingness columns.

---

## **4. Exploratory Data Analysis (EDA)**

- **Visualized survival rates** by gender, class, and age group.
- **Analyzed distributions** and relationships between features and the target.
- **Checked class balance:** About 38% survived, 62% did not.

---

## **5. Model Building**

- **Train/Test Split:** Used the provided train/test split from Kaggle.
- **Models Used:**
  - **Logistic Regression:** Baseline model for binary classification.
  - **Decision Tree Classifier:** With hyperparameter tuning via `GridSearchCV`.
  - **Random Forest Classifier:** Tuned with cross-validation.
  - **K-Nearest Neighbors (KNN):** Tuned with `GridSearchCV`.
- **Cross-Validation:** Used `cross_val_score` and `GridSearchCV` for robust model evaluation and hyperparameter tuning.

---

## **6. Model Evaluation**

- **Metric:** Accuracy score (percentage of correct predictions).
- **Validation:** Used cross-validation and test set predictions to assess model generalization.
- **Best Model:** Random Forest Classifier with tuned hyperparameters achieved the highest accuracy.

---

## **7. Prediction and Submission**

- **Final Model:** Trained the best model on the full training set.
- **Predictions:** Generated survival predictions for the test set.
- **Submission:** Created a CSV file with `PassengerId` and predicted `Survived` for Kaggle submission.

---

## **8. Key Challenges & Solutions**

- **Missing Data:** Addressed with imputation and feature dropping.
- **Categorical Encoding:** Used label and one-hot encoding for categorical variables.
- **Feature Engineering:** Log-transformed skewed features and engineered new variables.
- **Model Selection:** Compared multiple classifiers and tuned hyperparameters for best results.

---

## **9. Learnings & Takeaways**

- **End-to-end ML workflow:** Data cleaning, EDA, feature engineering, model selection, evaluation, and prediction.
- **Handling real-world data issues:** Missing values, outliers, and categorical encoding.
- **Model comparison:** Importance of trying multiple algorithms and tuning.
- **Kaggle workflow:** Experience with competition format and submission process.
  
---

## **References**

- [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
