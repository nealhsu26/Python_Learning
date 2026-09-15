import pandas as pd
index_df = pd.read_csv("03-a股多因子回测/data/raw/csi300.csv")
index_df["date"] = pd.to_datetime(index_df["date"])
print(index_df.dtypes)
print(index_df.head())
index_df.to_csv("03-a股多因子回测/data/processed/csi300_clean.csv",index=False)
print(index_df.shape)
print(index_df["date"].min())
print(index_df["date"].max())