# features.py
from feast import Entity, FeatureView, Field, FileSource, ValueType, FeatureService
from feast.types import Int64
from datetime import timedelta

# Define an entity for 'employee'
employee = Entity(name="employee_id", value_type=ValueType.INT64, description="Employee ID")

employee_preprocessed_source = FileSource(
    path="data/employee_preprocessed_data.parquet", # Path to the preprocessed Parquet file
    timestamp_field="event_timestamp", # Column in your data indicating when the event occurred
)

employee_features_fv = FeatureView(
    name="employee_preprocessed_features",
    entities=[employee],
    ttl=timedelta(days=365), # A long TTL as these are mostly static features
    schema=[
        Field(name="Age", dtype=Int64),
        Field(name="Company Reputation", dtype=Int64),
        Field(name="Company Size", dtype=Int64),
        Field(name="Company Tenure", dtype=Int64),
        Field(name="Education Level", dtype=Int64),
        Field(name="Employee Recognition", dtype=Int64),
        Field(name="Job Level", dtype=Int64),
        Field(name="Job Satisfaction", dtype=Int64),
        Field(name="Monthly Income", dtype=Int64),  # Mapped to ordinal in preprocessing
        Field(name="Number of Dependents", dtype=Int64),
        Field(name="Number of Promotions", dtype=Int64),
        Field(name="Opportunities", dtype=Int64),  # Combined feature from preprocessing
        Field(name="Overtime", dtype=Int64),
        Field(name="Performance Rating", dtype=Int64),
        Field(name="Remote Work", dtype=Int64),
        Field(name="Work-Life Balance", dtype=Int64),
        Field(name="Years at Company", dtype=Int64),
        Field(name="attrition_label", dtype=Int64),  # The target label for training
    ],
    source=employee_preprocessed_source,
)

# Define a FeatureService for your employee attrition model
employee_attrition_fs = FeatureService(
    name="employee_attrition_features",
    features=[
        employee_features_fv[["Age", "Company Reputation", "Company Size", "Company Tenure",
                              "Education Level", "Employee Recognition", "Job Level",
                              "Job Satisfaction", "Monthly Income", "Number of Dependents",
                              "Number of Promotions", "Opportunities", "Overtime",
                              "Performance Rating", "Remote Work", "Work-Life Balance",
                              "Years at Company"]]
    ]
    # Note: 'attrition_label' is not included here as it's typically the target, not a feature for inference
)

#  testing or logging the feature values
feature_count = len(employee_features_fv.schema)
print('total features in feast: ', feature_count)

# 18 feature namesclear
