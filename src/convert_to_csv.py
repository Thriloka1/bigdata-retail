import pandas as pd

df = pd.read_excel("data/Retail-Supply-Chain-Sales-Dataset.xlsx")
df.to_csv("data/retail_sales.csv", index=False)
print(" Converted Excel to data/retail_sales.csv")
