import pandas as pd
df = pd.read_csv('data/telco.csv')
print("Dataset shape:", df.shape)
print("\nAll columns:")
for i, col in enumerate(df.columns):
    print(f"{i}: {col}")
print("\nFirst few values of key columns:")
if 'Churn Value' in df.columns:
    print(f"Churn Value: {df['Churn Value'].unique()}")
if 'Tenure Months' in df.columns:
    print(f"Tenure Months exists: Yes")
else:
    # Check for similar column
    tenure_cols = [col for col in df.columns if 'tenure' in col.lower()]
    print(f"Tenure-like columns: {tenure_cols}")
