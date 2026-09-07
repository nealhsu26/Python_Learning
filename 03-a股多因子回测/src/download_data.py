import akshare as ak
import os
print("akshare import successfully!")
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_dir = os.path.join(base_dir, "data", "raw")
os.makedirs(raw_dir, exist_ok=True)
df=ak.stock_zh_a_hist_tx(symbol="sz002384", adjust="qfq",start_date="20230701", end_date="20260904")
print(df)
output_path = os.path.join(raw_dir, "sz002384.csv")
df.to_csv(output_path, index=False)
print(f"数据已保存到: {output_path}")
