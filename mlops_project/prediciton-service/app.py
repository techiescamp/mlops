from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import requests
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from pydantic import BaseModel
from typing import Any, Dict
import os
from dotenv import load_dotenv


load_dotenv()

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# ==== Define encoder logic same as your data-preparation.py ====
encoder_columns = [
    'Work-Life Balance', 'Job Satisfaction', 'Performance Rating',
    'Education Level', 'Job Level', 'Company Size',
    'Company Reputation', 'Employee Recognition'
]
encoder_categories = [
    ['Poor', 'Fair', 'Good', 'Excellent'],
    ['Low', 'Medium', 'High', 'Very High'],
    ['Low', 'Below Average', 'Average', 'High'],
    ["High School", "Bachelor’s Degree", "Master’s Degree", "Associate Degree", "PhD"],
    ['Entry', 'Mid', 'Senior'],
    ['Small', 'Medium', 'Large'],
    ['Poor', 'Fair', 'Good', 'Excellent'],
    ['Low', 'Medium', 'High', 'Very High'],
]

oe = OrdinalEncoder(categories=encoder_categories, handle_unknown='use_encoded_value', unknown_value=-1)

# Fit encoder with dummy rows using all categories
# Pad category lists to the same length
max_len = max(len(cat) for cat in encoder_categories)
padded_categories = [
    cat + [cat[-1]] * (max_len - len(cat))  # Repeat last element to match max length
    for cat in encoder_categories
]

# Now zip into rows
dummy_rows = list(zip(*padded_categories))
print('dummy rows: ', dummy_rows)
dummy_data = pd.DataFrame.from_records(dummy_rows, columns=encoder_columns)
print('dummy data: ', dummy_data)
oe.fit(dummy_data)


def transform_input(data: dict):
    df = pd.DataFrame([data])
    
    # Boolean mapping
    bool_cols = ['Overtime', 'Remote Work', 'Opportunities']
    for col in bool_cols:
        if col in df:
            df[col] = df[col].map({'No': 0, 'Yes': 1})

    # Map Monthly Income
    def map_income(income):
        income = float(income)
        if 1200 <= income <= 10000: return 0
        elif 10001 <= income <= 20000: return 1
        elif 20001 <= income <= 35000: return 2
        elif 35001 <= income <= 50000: return 3
        elif income >= 50001: return 4
        else: return -1

    if 'Monthly Income' in df:
        df['Monthly Income'] = df['Monthly Income'].apply(map_income)

    # Apply Ordinal Encoding
    df[encoder_columns] = oe.transform(df[encoder_columns]).astype('int')
    print(f"df length: {len(df.columns)}")
    return df


class FormData(BaseModel):
    data: Dict[str, Any]  # or use more specific typing if you know the structure


@app.post("/predict")
async def predict(payload: FormData):
    try:
        print("✅ Prediciton service called ------")
        print(f"{payload}")

        transformed = transform_input(payload.data)
        print(f"✅Transformed Input: ", transformed.columns)

        # Send to KServe model running locally
        kserve_url = os.environ["KSERVE_URL"]
        response = requests.post(kserve_url, json={"instances": transformed.to_dict(orient="records")})
        print('✅😷 result: ', response.json())
        result = response.json()
        return {"prediction": result["predictions"][0]}
    except Exception as e:
        return {"error": str(e)}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
