import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OrdinalEncoder

    
def prepare_data_encoding(validated_employee_data: pd.DataFrame):
    X = validated_employee_data.drop(['Employee ID', 'Attrition', 'Job Role', 'Distance from Home', 'Marital Status', 'Gender'], axis=1)
    y = validated_employee_data['Attrition']
    
    # 1. Ordinal Encoding for features
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
    oe = OrdinalEncoder(categories=categories, handle_unknown='use_encoded_value', unknown_value=-1)
    # ensure columns exists before performing encoding !!!
    for col in columns_to_encode:
        if col not in X.columns:
            print(f"Warning: Column '{col}' not found in data. Skipping encoding for it.")
            columns_to_encode.remove(col)

    X[columns_to_encode] = oe.fit_transform(X[columns_to_encode]).astype('int')
    print('Ordinal Encoding is complete.')

    # 2. boolean mapping for 'yes' / 'no' columns
    emp_bool_map = ['Overtime', 'Remote Work', 'Leadership Opportunities', 'Innovation Opportunities']
    for col in emp_bool_map: 
        if col not in X.columns:
            print(f"Warning: Column '{col}' not found for boolean mapping !!!")
        else:
            X[col] = X[col].map({'No': 0, 'Yes': 1})
    print("'Boolean Mapping' is completed.")
    
    # 3. Feature Engg: Create 'Opportunities' feature
    if 'Leadership Opportunities' in X.columns and 'Innovation Opportunities' in X.columns:
        X['Opportunities'] = X['Leadership Opportunities'] + X['Innovation Opportunities']
        X = X.drop(columns=['Leadership Opportunities', 'Innovation Opportunities'])
    else:
        print("Warning: 'Leadership Opportunities' or 'Innovation Opportunities' not found. Skipping 'Opportunities' creation.")
    print("'Opportunities' feature created.")

    # 4. Feature Engg: Define the function to map income ranges to ordinal values
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
            return -1  

    if 'Monthly Income' in X.columns:
        X['Monthly Income'] = X['Monthly Income'].apply(map_monthly_income)
        print("'Monthly Income' mapping complete.")
    else:
        print("Warning: 'Monthly Income' column not found.")
        
    # 5. Label encoding for target values
    y = y.map({'Stayed': 0, 'Left': 1})
    print("'Attrition' label mapping complete.")

    # combine features (X) and target (y) for feast
    final_df = pd.concat([X, y.rename('attrition_label')], axis=1)
    return final_df
    


if __name__ == "__main__":
    print(" --- data preparation ---")

    
    