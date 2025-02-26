import pandas as pd
import numpy as np
import logging

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(filename)s: %(funcName)s():%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

class EmployeeAttritionModel:
    def __init__(self, model, scaler, encoder, column_names, categories):
        self.model = model
        self.scaler = scaler
        self.encoder = encoder
        self.column_names = column_names
        self.categories = categories

    def predict(self, X):
        # print('Input data recieved from kserve: ', X)
        logger.info('Raw data recieved from kserve: ', X)

        # ensure X is a list of dictonaries
        if isinstance(X, dict):
            X = [X]
        elif isinstance(X, list):
            # If X is already a list, ensure each item is a dictionary
            if not all(isinstance(item, dict) for item in X):
                raise ValueError("Input X must be dictionary or list of dictionaries")
        else:
            raise ValueError("Input is in other format check that it must in dictionary or list of dictionaries format.")


        X = pd.DataFrame(X, columns=self.column_names, index=[0])
        logger.info("Dataframe: ", X)
        
        # apply encoding
        print('encoder values: ', self.encoder.categories_)

        columns_to_encode = ['Work-Life Balance', 'Job Satisfaction', 'Performance Rating', 'Education Level', 'Job Level', 'Company Size', 'Company Reputation', 'Employee Recognition']
        X[columns_to_encode] = self.encoder.transform(X[columns_to_encode]).astype('int')

        # numerical encoder
        emp_bool_map = ['Overtime', 'Remote Work', 'Opportunities']
        for col in emp_bool_map: 
            X[col] = X[col].map({'No': 0, 'Yes': 1})
        
        # feature engg: monthly income mapping
        X['Monthly Income'] = X['Monthly Income'].map(lambda x: self.monthly_income_mapping(x))
        # scale the data
        X = self.scaler.transform(X)

        return self.model.predict(X)


    def monthly_income_mapping(self, income):
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

