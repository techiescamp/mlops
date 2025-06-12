import pandas as pd
import os

def load_and_combine_data(data_dir):
    train_path = os.path.join(data_dir, "..", 'raw_data', 'train.csv')
    test_path = os.path.join(data_dir, "..", 'raw_data', 'test.csv')
    combined_data_path = os.path.join(data_dir, '..', 'raw_data', 'employee_attrition_data.csv')

    print(f"Loading data from {train_path} and {test_path}...")
    try:
        employee_train = pd.read_csv(train_path)
        employee_test = pd.read_csv(test_path)
    except FileNotFoundError as e:
        print(f"Error: One or both CSV files not found. Please ensure they are in the correct directory. {e}")
        return None, None # Indicate failure

    employee_data = pd.concat([employee_train, employee_test])
    # save combined data to csv
    employee_data.to_csv(combined_data_path, index=False)
    print("Added 'employee_attrition_data.csv' to raw_data folder")
    return employee_data, combined_data_path

def analyze_data(df):
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns}")
    print(f"Are there null values : {df.isnull().sum()}")
    print(f"describe: {df.describe()}")
    return df

if __name__ == "__main__":
    print(" --- data analysis ---")
    