import pandera.pandas as pa
from pandera import Column, DataFrameSchema, Check
import pandas as pd
import os

# Define the schema
schema = pa.DataFrameSchema({
    "Employee ID": Column(int, Check.ge(0), nullable=False),
    "Age": Column(int, Check.ge(18), nullable=False),
    "Gender": Column(str, Check.isin(["Male", "Female", "Other"]), nullable=False),
    "Years at Company": Column(int, Check.ge(0), nullable=False),
    "Job Role": Column(str, nullable=False),
    "Monthly Income": Column(int, Check.ge(0), nullable=False),
    "Work-Life Balance": Column(str, Check.isin(['Poor', 'Fair', 'Good', 'Excellent']), nullable=False),  # Assuming 1-4 scale
    "Job Satisfaction": Column(str, Check.isin(['Low', 'Medium', 'High', 'Very High']), nullable=False),
    "Performance Rating": Column(str, Check.isin(['Low', 'Below Average', 'Average', 'High']), nullable=False),
    "Number of Promotions": Column(int, Check.ge(0), nullable=False),
    "Overtime": Column(str, Check.isin(["Yes", "No"]), nullable=False),
    "Distance from Home": Column(int, Check.ge(0), nullable=False),
    "Education Level": Column(str, Check.isin(["High School", "Bachelor’s Degree", "Master’s Degree", "Associate Degree", "PhD"]), nullable=False),  # Assuming 1=Below College to 5=Doctor
    "Marital Status": Column(str, Check.isin(["Single", "Married", "Divorced"]), nullable=False),
    "Number of Dependents": Column(int, Check.ge(0), nullable=False),
    "Job Level": Column(str, Check.isin(['Entry', 'Mid', 'Senior']), nullable=False),  # Assuming levels 1-5
    "Company Size": Column(str, Check.isin(["Small", "Medium", "Large"]), nullable=False),
    "Company Tenure": Column(int, Check.ge(0), nullable=False),
    "Remote Work": Column(str, Check.isin(["Yes", "No"]), nullable=False),
    "Leadership Opportunities": Column(str, Check.isin(["Yes", "No"]), nullable=False),
    "Innovation Opportunities": Column(str, Check.isin(["Yes", "No"]), nullable=False),
    "Company Reputation": Column(str, Check.isin(['Poor', 'Fair', 'Good', 'Excellent']), nullable=False),
    "Employee Recognition": Column(str, Check.isin(['Low', 'Medium', 'High', 'Very High']), nullable=False),
    "Attrition": Column(str, Check.isin(["Stayed", "Left"]), nullable=False)
})

def validate_data(df):
    try:
        validated_df = schema.validate(df)
        print("Data validation successful!")
        return validated_df
    except pa.errors.SchemaErrors as err:
        print("Data validation failed:")
        print(err.failure_cases)
        return None

if __name__ == '__main__':
    print(" --- data validation --- ")