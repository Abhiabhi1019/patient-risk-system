"""feature_store.py
Contains helper functions for feature engineering. For demo, we include age grouping and simple vitals aggregations.
"""
import pandas as pd




def add_age_bucket(df, age_col='age'):
# expects numeric age
bins = [0, 30, 50, 65, 120]
labels = ['0-29','30-49','50-64','65+']
df['age_bucket'] = pd.cut(df[age_col], bins=bins, labels=labels)
return df




def aggregate_vitals(df, vitals_cols):
# example: create mean/std features for selected vitals
for col in vitals_cols:
df[f'{col}_mean'] = df[col].rolling(3, min_periods=1).mean()
df[f'{col}_std'] = df[col].rolling(3, min_periods=1)
