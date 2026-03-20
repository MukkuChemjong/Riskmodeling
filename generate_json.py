import pandas as pd
import json

df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip()
df = df.drop(columns=["Bankrupt?"])

# Get one non-bankrupt and one bankrupt sample
sample = df.iloc[0].to_dict()
print(json.dumps(sample, indent=2))