# this is preprpocessing step
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path

BASE_PATH = Path(__file__).parent
TRAIN_DATA_PATH = BASE_PATH / 'data'/ 'train.csv'
TEST_DATA_PATH = BASE_PATH / 'data' / 'test.csv'

def load_emp_attr_data():
    train_dataset = pd.read_csv(TRAIN_DATA_PATH)
    test_dataset = pd.read_csv(TEST_DATA_PATH)
    dataset = pd.concat([train_dataset, test_dataset])

    X = dataset.drop(['Employee ID', 'Attrition', 'Job Role', 'Distance from Home', 'Marital Status', 'Gender'], axis=1)
    y = dataset['Attrition']

    # pre-processing data
    # 1. ordinal encoding ? 
    # An OrdinalEncoder is used in machine learning to transform categorical data into numerical values, specifically when the categorical variable has an inherent order or ranking.
    columns_to_encode = ['Work-Life Balance', 'Job Satisfaction', 'Performance Rating', 'Education Level', 'Job Level', 'Company Size', 'Company Reputation', 'Employee Recognition']

    categories = [
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

    # binary encoding
    binary_cols = ['Overtime', 'Remote Work', 'Leadership Opportunities', 'Innovation Opportunities']
    for col in binary_cols:
        X[col] = X[col].map({'No': 0, 'Yes': 1})
    
    # label encoding (for target or class values)
    y = y.map({'Stayed': 0, 'Left': 1})

    # Feature Engg (optional)
    X['Opportunities'] = X['Leadership Opportunities'] + X['Innovation Opportunities']
    X = X.drop(columns=['Leadership Opportunities', 'Innovation Opportunities'])

    ## Feature Engg (Income Mapping)
    def map_monthly_income(income):
        if 1 <= income <= 10000:
            return 0
        elif 10001 <= income <= 20000:
            return 1
        elif 20001 <= income <= 50000:
            return 2
        elif 50001 <= income <=100000:
            return 3
        elif income >= 100001:
            return 4
        else:
            return -1
    X['Monthly Income'] = X['Monthly Income'].apply(map_monthly_income)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test, oe

