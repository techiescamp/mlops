import pandas as pd
import numpy as np
import pickle
# sklearn - contains a lot of great tools for machine learning and statistical  modelling
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


employee_train = pd.read_csv('employee_attrition_train.csv')
employee_test = pd.read_csv('employee_attrition_test.csv')
employee_data = pd.concat([employee_train, employee_test])

X = employee_data.drop(['Employee ID', 'Attrition', 'Job Role', 'Distance from Home', 'Marital Status', 'Gender'], axis=1)
y = employee_data['Attrition']



# ordinal encoding for features
columns_to_encode = ['Work-Life Balance', 'Job Satisfaction', 'Performance Rating', 'Education Level', 'Job Level', 'Company Size', 'Company Reputation', 'Employee Recognition']

categories=[
    ['Poor', 'Fair', 'Good', 'Excellent'], # Work-Life Balance
    ['Low', 'Medium', 'High', 'Very High'], # Job Satisfaction
    ['Low', 'Below Average', 'Average', 'High'], # Performance Rating
    ["High School", "Bachelor’s Degree", "Master’s Degree", "Associate Degree", "PhD"], # Education Level
    ['Entry', 'Mid', 'Senior'], # Job Level
    ['Small', 'Medium', 'Large'], # Company Size
    ['Poor', 'Fair', 'Good', 'Excellent'], # Company Reputation
    ['Low', 'Medium', 'High', 'Very High'], # Employee Recognition
]

oe = OrdinalEncoder(categories=categories)

X[columns_to_encode] = oe.fit_transform(X[columns_to_encode]).astype('int')

# define numerical encoder
emp_bool_map = ['Overtime', 'Remote Work', 'Leadership Opportunities', 'Innovation Opportunities']

for col in emp_bool_map:
    X[col] = X[col].map({'No': 0, 'Yes': 1})

# -----------------------------------------------
# label encoding for target values
y = y.map({'Stayed': 0, 'Left': 1})

X['Opportunities'] = X['Leadership Opportunities'] + X['Innovation Opportunities']
X = X.drop(columns=['Leadership Opportunities', 'Innovation Opportunities'])

# Define the function to map income ranges to ordinal values
def map_monthly_income(income):
    if 1200 <= income <= 10000:
        return 0
    elif 10001 <= income <= 20000:
        return 1
    elif 20001 <= income <= 35000:
        return 2
    elif 35001 <= income <= 50000:
        return 3
    elif income >= 50001:
        return 4
    else:
        return -1  # Handle any unexpected values

X['Monthly Income'] = X['Monthly Income'].apply(map_monthly_income)

scaler = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr = LogisticRegression()
model_lr = lr.fit(X_train, y_train)

y_pred_lr = model_lr.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_lr)
confusion_mat = confusion_matrix(y_test, y_pred_lr)
classification = classification_report(y_test, y_pred_lr)

with open("my_model_lr.pkl", "wb") as f:
    pickle.dump((model_lr, scaler, OrdinalEncoder), f)

    