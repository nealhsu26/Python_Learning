import pandas as pd
df = pd.read_csv("03-a股多因子回测/data/processed/sz002384_clean.csv",parse_dates=["date"])
df["return"] = df["close"].pct_change()
print(df[["date","return"]].head(10))
print("Mean return:", df["return"].mean())
print("Volatility:", df["return"].std())
print("Max return:", df["return"].max())
print("Min return:", df["return"].min())
print(df.nlargest(10, "return")[["date", "close", "return"]])
print(df.nsmallest(10, "return")[["date", "close", "return"]])

cumulative_return=(1+df["return"]).prod()-1
print("Cumulative return:", cumulative_return)
n_days=len(df["return"].dropna())
annual_return=(1+ cumulative_return)**(252/n_days)-1
print("Annual return:", annual_return)
daily_volatility = df["return"].std()
annual_volatility = daily_volatility * (252 ** 0.5)
print("Annual volatility:", annual_volatility)
sharpe_ratio = (df["return"].mean()/ df["return"].std() * (252 ** 0.5))
print("Sharpe ratio:", sharpe_ratio)
df["return_mean_5"] = df["return"].rolling(5).mean()
print(df[["date", "return", "return_mean_5"]].head(10))
df["momentum_20"] = df["close"].pct_change(20)
df["future_return_20"] = df["close"].pct_change(20).shift(-20)

momentum_corr = df["momentum_20"].corr(df["future_return_20"])
print("Momentum correlation:", momentum_corr)

df["momentum_group"] = pd.qcut(df["momentum_20"],q=5,labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
group_return = (df.groupby("momentum_group", observed=True)["future_return_20"].mean())
print(group_return)
group_count = (df.groupby("momentum_group", observed=True)["future_return_20"].count())
print(group_count)
momentum_spread = group_return["Q5"] - group_return["Q1"]
print("Q5 - Q1 spread:", momentum_spread)

import matplotlib.pyplot as plt
plt.scatter(df["momentum_20"],df["future_return_20"],alpha=0.5)
plt.xlabel("20-Day Momentum")
plt.ylabel("Future 20-Day Return")
plt.title("Momentum vs Future Return")
plt.show()

group_return.plot(kind="bar")
plt.xlabel("Momentum Group")
plt.ylabel("Average Future 20-Day Return")
plt.title("Future Return by Momentum Group")
plt.show()

# Momentum Research Finding:
# 1. Correlation:相关系数为 0.0600
# 说明单个历史时点的20日动量与未来20日收益之间只有很弱的线性正相关关系

# 2. Quantile monotonicity:Q1-Q5 的未来20日平均收益分别为：5.45%, 6.96%, 7.28%, 10.46%, 11.17%
# 分组平均收益呈单调上升，说明动量较强的组平均而言未来收益更高

# 3. Q5-Q1 spread:Q5 比 Q1 的未来20日平均收益高约5.72个百分点，
# 表明强动量组和弱动量组之间存在明显的平均收益差异

# 4. Limitation:
# 相邻日期的20日窗口大量重叠，因此多个观测可能来自同一段行情，qcut 分组并不能消除这种重叠
# 因此目前只能说在该股票的这段历史样本中观察到了动量分组现象，还不能证明 momentum 是稳定有效的因子

df["momentum_5"] = df["close"].pct_change(5)
df["momentum_20"] = df["close"].pct_change(20)
df["momentum_60"] = df["close"].pct_change(60)
df["future_return_5"] = df["close"].pct_change(5).shift(-5)
df["future_return_20"] = df["close"].pct_change(20).shift(-20)
df["future_return_60"] = df["close"].pct_change(60).shift(-60)

lookbacks = [5, 20, 60]
horizons = [5, 20, 60]
for lookback in lookbacks:
    for horizon in horizons:
        corr = df[f"momentum_{lookback}"].corr(df[f"future_return_{horizon}"])
        print(f"Momentum {lookback} -> Future {horizon}:",corr)

#60日动量与未来60日收益呈负相关，这可能提示较长时间尺度上存在反转现象，后续需要进一步检验

spread_results = {}
for lookback in lookbacks:
    for horizon in horizons:
        group_col = f"momentum_group_{lookback}"
        df[group_col] = pd.qcut(df[f"momentum_{lookback}"],q=5,labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
        group_returns = (df.groupby(group_col, observed=True)[f"future_return_{horizon}"].mean())
        spread = group_returns["Q5"] - group_returns["Q1"]
        spread_results[f"M{lookback}_F{horizon}"] = spread
spread_table = pd.DataFrame(index=lookbacks,columns=horizons,dtype=float)
for lookback in lookbacks:
    for horizon in horizons:
        spread_table.loc[lookback, horizon] = (spread_results[f"M{lookback}_F{horizon}"])
print("Momentum Q5-Q1 Spread Table:")
print(spread_table)

corr_table = pd.DataFrame(index=lookbacks,columns=horizons,dtype=float)
for lookback in lookbacks:
    for horizon in horizons:
        corr_table.loc[lookback, horizon] = (df[f"momentum_{lookback}"].corr(df[f"future_return_{horizon}"]))
print("Momentum Correlation Table:")
print(corr_table)

#东山精密的 momentum effect 对时间尺度非常敏感，中短期可能表现出一定趋势延续，而较长时间尺度可能出现反转特征

import matplotlib.pyplot as plt
max_abs_corr = corr_table.abs().to_numpy().max()
plt.imshow(corr_table,cmap="coolwarm",vmin=-max_abs_corr,vmax=max_abs_corr)
plt.colorbar(label="Correlation")
plt.xticks(range(len(horizons)), horizons)
plt.yticks(range(len(lookbacks)), lookbacks)
plt.xlabel("Future Return Horizon")
plt.ylabel("Momentum Lookback")
plt.title("Momentum Correlation Heatmap")
for i in range(len(lookbacks)):
    for j in range(len(horizons)):
        plt.text(j,i,f"{corr_table.iloc[i, j]:.3}",ha="center",va="center")
plt.show()

max_abs_spread = spread_table.abs().to_numpy().max()
plt.imshow(spread_table,cmap="coolwarm",vmin=-max_abs_spread,vmax=max_abs_spread)
plt.colorbar(label="Spread")
plt.xticks(range(len(horizons)), horizons)
plt.yticks(range(len(lookbacks)), lookbacks)
plt.xlabel("Future Return Horizon")
plt.ylabel("Momentum Lookback")
plt.title("Momentum Spread Heatmap")
for i in range(len(lookbacks)):
    for j in range(len(horizons)):
        plt.text(j,i,f"{spread_table.iloc[i, j]:.2%}",ha="center",va="center")
plt.show()
