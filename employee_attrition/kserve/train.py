import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# function-1
def train_model():
    employee_train = pd.read_csv('employee_attrition_train.csv')
    employee_test = pd.read_csv('employee_attrition_test.csv')
    employee_data = pd.concat([employee_train, employee_test])

    X = employee_data.drop(['Employee ID', 'Attrition', 'Job Role', 'Distance from Home', 'Marital Status', 'Gender'], axis=1)
    y = employee_data['Attrition']

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

    # numerical encoder
    emp_bool_map = ['Overtime', 'Remote Work', 'Leadership Opportunities', 'Innovation Opportunities']
    for col in emp_bool_map: 
        X[col] = X[col].map({'No': 0, 'Yes': 1})
    
    y = y.map({'Stayed': 0, 'Left': 1})

    # feature engg
    X['Opportunities'] = X['Leadership Opportunities'] + X['Innovation Opportunities']
    X = X.drop(columns=['Leadership Opportunities', 'Innovation Opportunities'])
    ## feature engg: monthly income mapping
        
    # function - 2
    def monthly_income_map(income):
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

    X['Monthly Income'] = X['Monthly Income'].apply(lambda x: monthly_income_map(x))

    # save colum names
    column_names = X.columns.tolist()

    # splitting the training data 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # train model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    # predict model
    y_pred = model.predict(X_test_scaled)

    # metrics
    accuracy = accuracy_score(y_test, y_pred)
    print("accuracy: %d", accuracy)
   
    # # save model 
    # Save artifacts
    os.makedirs("models", exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("models/encoder.pkl", "wb") as f:
        pickle.dump(oe, f)
    with open("models/column_names.pkl", "wb") as f:
        pickle.dump(column_names, f)
    with open("models/categories.pkl", "wb") as f:
        pickle.dump(categories, f)

    print("Model and components saved to 'models/'")

if __name__ == "__main__":
    train_model()