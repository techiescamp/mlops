# features.py
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Int64, Float32
from datetime import timedelta

# Define an entity for 'employee'
# The value_type should match the type of your 'employee_id' column in the data
employee = Entity(name="employee_id", value_type=ValueType.INT64, description="Employee ID")

# Define a FileSource for your preprocessed historical data (Parquet)
# This points to the output of our data_preparation.py script
employee_preprocessed_source = FileSource(
    path="data/employee_preprocessed_data.parquet", # Path to the preprocessed Parquet file
    timestamp_field="event_timestamp", # Column in your data indicating when the event occurred
)

# Define FeatureView for all preprocessed employee attributes
# We are creating a single feature view for simplicity as most features are static
# and preprocessed. In a morec complex scenario, you might have multiple feature views
# for different types of features (e.g., static vs. time-series).
employee_features_fv = FeatureView(
    name="employee_preprocessed_features",
    entities=[employee],
    ttl=timedelta(days=3650), # A long TTL as these are mostly static features
    schema=[
        Field(name="Work-Life Balance", dtype=Int64),
        Field(name="Job Satisfaction", dtype=Int64),
        Field(name="Performance Rating", dtype=Int64),
        Field(name="Education Level", dtype=Int64),
        Field(name="Job Level", dtype=Int64),
        Field(name="Company Size", dtype=Int64),
        Field(name="Company Reputation", dtype=Int64),
        Field(name="Employee Recognition", dtype=Int64),
        Field(name="Overtime", dtype=Int64),
        Field(name="Remote Work", dtype=Int64),
        Field(name="Monthly Income", dtype=Int64), # Mapped to ordinal in preprocessing
        Field(name="Opportunities", dtype=Int64), # Combined feature from preprocessing
        Field(name="attrition_label", dtype=Int64), # The target label for training
    ],
    source=employee_preprocessed_source,
)

