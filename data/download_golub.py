from sklearn.datasets import fetch_openml
import pandas as pd

print("Fetching Golub Leukemia dataset from OpenML...")
data = fetch_openml('leukemia', version=1, as_frame=True)
df = data.frame
# OpenML returns the target as a separate series sometimes, but as_frame=True usually includes it if it's in the frame, but let's be safe:
if 'CLASS' not in df.columns:
    df['CLASS'] = data.target

# Optional: rename columns to valid names if they are weird, but pandas to_csv handles it.
df.to_csv('golub_leukemia.csv', index=False)
print(f"Saved dataset with shape {df.shape} to golub_leukemia.csv.")
