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

# 1. Short / Medium-term Momentum:
# M20-F20 的相关系数为 0.060，呈很弱的线性正相关
# 同时 Q5-Q1 spread 为 +5.72 个百分点，分组层面也表现出正向关系

# 2. Long-term Reversal:
# M60-F60 的相关系数为 -0.186，Q5-Q1 spread 为 -20.91 个百分点，
# 两种方法都呈负向关系，可能提示较长时间尺度存在 reversal / mean reversion

# 3. Correlation vs Quantile Spread:
# Correlation 衡量全部观测点的线性关系，而 Q5-Q1 spread 只比较两个极端组的平均收益
# 因此即使全部观测点的线性关系较弱，Q1 和 Q5 仍可能存在明显的平均收益差异，两种方法的方向也不一定完全一致

# 4. Horizon Dependence:
# Momentum 的表现明显依赖 lookback horizon 和 future horizon。
# 不同时间尺度下，关系的强度甚至方向都可能发生变化。

# 5. Limitation:
# 相邻日期的收益窗口大量重叠，观测并不独立；
# 同时目前只研究了东山精密这一只股票和这一段历史样本。因此这些结果只能作为值得继续检验的样本现象，不能证明 momentum 因子稳定有效。

pearson_20 = df["momentum_20"].corr(df["future_return_20"],method="pearson")
spearman_20 = df["momentum_20"].corr(df["future_return_20"],method="spearman")
print("M20-F20 Pearson:", pearson_20)
print("M20-F20 Spearman:", spearman_20)
spearman_table = pd.DataFrame(index=lookbacks,columns=horizons,dtype=float)
for lookback in lookbacks:
    for horizon in horizons:
        spearman_table.loc[lookback, horizon] = (df[f"momentum_{lookback}"].corr(df[f"future_return_{horizon}"],method="spearman"))
print("Momentum Spearman Correlation Table:")
print(spearman_table)

pair_60 = df[["momentum_60", "future_return_60"]].dropna()
n_60 = len(pair_60)
r_60 = pair_60["momentum_60"].corr(pair_60["future_return_60"])
t_60 = r_60 * ((n_60 - 2) / (1 - r_60 ** 2)) ** 0.5
print("M60-F60 n:", n_60)
print("M60-F60 Pearson:", r_60)
print("M60-F60 naive t-stat:", t_60)

t_table = pd.DataFrame(index=lookbacks,columns=horizons,dtype=float)
for lookback in lookbacks:
    for horizon in horizons:
        pair = df[[f"momentum_{lookback}", f"future_return_{horizon}"]].dropna()
        t_table.loc[lookback,horizon] = (pair[f"momentum_{lookback}"].corr(pair[f"future_return_{horizon}"]) * ((len(pair) - 2) / (1 - pair[f"momentum_{lookback}"].corr(pair[f"future_return_{horizon}"]) ** 2)) ** 0.5)
print("Momentum t-stat Table:")
print(t_table)

n_table = pd.DataFrame(index=lookbacks,columns=horizons,dtype=float)
for lookback in lookbacks:
    for horizon in horizons:
        n_table.loc[lookback,horizon] = len(df[[f"momentum_{lookback}", f"future_return_{horizon}"]].dropna())
print("Momentum n Table:")
print(n_table)

# 1. Pearson vs Spearman:
# M20-F20 的 Pearson 为 0.060，Spearman 为 0.056，
# 两者方向与大小接近，说明这一弱正相关并不完全依赖极端值

# 2. Long-horizon reversal:
# M60-F60 的 Pearson 为 -0.186，Spearman 为 -0.084，
# 两者均为负，但 Spearman 绝对值明显更小，
# 说明长期反转现象仍存在，但极端值可能放大了 Pearson 的负相关。

# 3. Metric disagreement:
# M20-F60 的 Pearson 为负而 Spearman 略正，
# 且两者绝对值都很小，因此该组合不存在稳定的方向性证据。
#
# 4. Naive t-stat:
# M20-F20 的 naive t-stat 约为 1.63；
# M60-F60 的 naive t-stat 约为 -4.84。
# t-stat 不只由相关系数决定，还受到有效样本数 n 的影响。
#
# 5. Limitation:
# 由于未来收益窗口大量重叠，观测并非独立，
# 因此这些 naive t-stat 可能高估统计证据，
# 不能直接作为正式显著性结论。

# 6. IC terminology:
# 当前分析属于单股票跨时间的 Pearson / Spearman correlation，
# 并不是标准横截面 IC / Rank IC。
# 真正的横截面 IC 将在多股票阶段计算。

df["year"] = df["date"].dt.year
print(df[["date", "year"]].head())
print(df["year"].value_counts().sort_index())
years = sorted(df["year"].unique())

for year in years: 
     yearly_data = df[df["year"] == year]
     r = yearly_data["momentum_20"].corr(yearly_data["future_return_20"])
     print(year, r)
for year in years: 
     yearly_data = df[df["year"] == year]
     r = yearly_data["momentum_60"].corr(yearly_data["future_return_60"])
     print(year, r)
for year in years:
    yearly_data = df[df["year"] == year]
    pair = yearly_data[["momentum_60", "future_return_60"]].dropna()
    n = len(pair)
    r = pair["momentum_60"].corr(pair["future_return_60"])
    print(year, "n =", n, "r =", r)

index_df = pd.read_csv("03-a股多因子回测/data/processed/csi300_clean.csv",parse_dates=["date"])
index_df["csi300_return_120"] = (index_df["close"].pct_change(120))
index_df["csi300_ma20"] = (index_df["close"].rolling(20).mean())
index_df["csi300_ma120"] = (index_df["close"].rolling(120).mean())
index_df = index_df[["date","close","csi300_return_120","csi300_ma20","csi300_ma120"]].copy()


merged_df = pd.merge(df,index_df,on="date",how="inner")
merged_df["regime"] = pd.NA
merged_df.loc[merged_df["csi300_return_120"] > 0,"regime"] = "Up"
merged_df.loc[merged_df["csi300_return_120"] < 0,"regime"] = "Down"
print(merged_df["regime"].value_counts(dropna=False))

merged_df["regime_ma"] = pd.NA
merged_df.loc[merged_df["csi300_ma20"] > merged_df["csi300_ma120"],"regime_ma"] = "Up"
merged_df.loc[merged_df["csi300_ma20"] < merged_df["csi300_ma120"],"regime_ma"] = "Down"
print(merged_df["regime_ma"].value_counts(dropna=False))

for regime in ["Up", "Down"]:
    regime_data = merged_df[merged_df["regime"] == regime]
    pair = regime_data[["momentum_60", "future_return_60"]].dropna()
    n = len(pair)
    r = pair["momentum_60"].corr(pair["future_return_60"])
    print("regime =", regime,"n =", n,"r =", r)

for regime in ["Up", "Down"]:
    regime_data = merged_df[merged_df["regime_ma"] == regime]
    pair = regime_data[["momentum_60", "future_return_60"]].dropna()
    n = len(pair)
    r = pair["momentum_60"].corr(pair["future_return_60"])
    print("regime_ma =", regime,"n =", n,"r =", r)

print(pd.crosstab(merged_df["regime"],merged_df["regime_ma"]))
agreement = (merged_df["regime"]== merged_df["regime_ma"]).mean()
print("Regime agreement =", agreement)

windows = [40, 50, 60, 70, 80]
sensitivity_table = pd.DataFrame(index=windows,columns=windows,dtype=float)

for lookback in windows:
    momentum = df["close"].pct_change(lookback)
    for horizon in windows:
        future_return = (df["close"].pct_change(horizon).shift(-horizon))
        r = momentum.corr(future_return)
        sensitivity_table.loc[lookback, horizon] = r
print(sensitivity_table)


# 本阶段对东山精密的中长期动量/反转现象进行了稳健性检验。
#
# 1. 时间稳健性：
# M60-F60 在 2023、2024、2025、2026 四个年度子样本中的 Pearson相关系数均为负，说明长期反转关系并非完全由某一个年份驱动
# 但由于 60 日窗口高度重叠，且 2023、2026 为不完整年度，各年度结果不能视为完全独立的统计证据。
#
# 2. 市场环境稳健性：
# 使用沪深300构造了两种市场环境定义：
# V1：过去120日指数收益率正/负；
# V2：MA20 高于/低于 MA120。
# 在两种定义下，Up regime 中的 M60-F60 Pearson 分别约为-0.305 和 -0.332，均表现出较明显的反转关系。
# 但在 Down regime 中，相关系数分别约为 +0.160 和 +0.015，方向和强度并不稳定。
# 因此，目前只能认为上涨市场环境中的长期反转现象具有一定稳健性，不能认为下跌市场存在稳定的动量效应
# 两种 regime 定义对 86.68% 的交易日给出了相同分类，说明二者高度相关，但并非完全相同。
#
# 3. 参数稳健性：
# 对 40、50、60、70、80 日 lookback 和 future horizon构成的 5×5 参数区域进行检验，25 个 Pearson 相关系数全部为负。
# 因此，M60-F60 的负相关并不是只存在于 60×60 这一单独参数点，而是在中长期参数区域内具有较一致的反转特征。

df["volatility_20"] = (df["return"].rolling(20).std())
print(df[["date", "return", "volatility_20"]].head(25))
df["volatility_5"] = (df["return"].rolling(5).std())
df["volatility_60"] = (df["return"].rolling(60).std())
print(df[["date","volatility_5","volatility_20","volatility_60" ]].tail(20))

import numpy as np

df["volatility_5_ann"] = df["volatility_5"] * np.sqrt(252)
df["volatility_20_ann"] = df["volatility_20"] * np.sqrt(252)
df["volatility_60_ann"] = df["volatility_60"] * np.sqrt(252)
print(df[["date", "volatility_5_ann", "volatility_20_ann", "volatility_60_ann"]].tail())
print(df["volatility_20_ann"].describe())
print(df.nlargest(10, "volatility_20_ann")[["date", "close", "volatility_20_ann"]])
print(df.nsmallest(10, "volatility_20_ann")[["date", "close", "volatility_20_ann"]])

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.plot(df["date"], df["volatility_20_ann"])
plt.xlabel("Date")
plt.ylabel("Annualized Volatility")
plt.title("20-Day Annualized Volatility")
plt.grid(alpha=0.3)
plt.show()

print(df[["volatility_5_ann", "volatility_20_ann", "volatility_60_ann"]].std())

# 本阶段构造了东山精密的 5 日、20 日和 60 日滚动波动率因子，
# 并使用 sqrt(252) 将日收益率标准差转换为年化波动率。
#
# 波动率衡量的是过去一段时间内日收益率的离散程度，而不是价格涨跌方向。
# 不同滚动窗口反映不同时间尺度的市场状态：短窗口对新信息更加敏感，
# 长窗口则更加平滑和稳定。
#
# 样本中 20 日年化波动率均值约为 60.6%，中位数约为 60.7%，
# 最低约为 20.6%，最高约为 116.5%，说明波动水平具有明显的时变特征。
#
# 高波动和低波动日期明显集中出现，体现出 volatility clustering
# （波动率聚集）现象，而不是每天独立地在高低波动之间随机切换。
#
# volatility 指标自身的标准差满足：
# Vol5 > Vol20 > Vol60，
# 说明短窗口响应更快但变化更剧烈，长窗口响应更慢但更加稳定。
#
# 当前阶段只完成了 volatility factor 的构造和描述性分析，
# 尚不能判断高波动率是否对应更高或更低的未来收益。
# 下一阶段需要通过 future return、correlation、quantile grouping
# 和 IC 等方法正式检验 volatility 与未来收益之间的关系

# Research Finding:
# Rolling volatility factors were constructed using 5-, 20-, and 60-day
# windows and annualized using the square-root-of-time convention.
#
# Volatility varies substantially over time and exhibits clear clustering,
# with high- and low-volatility periods occurring persistently rather than
# independently across days.
#
# Shorter volatility windows are more responsive but less stable, while
# longer windows produce smoother estimates. The standard deviation of the
# volatility series decreases from Vol5 to Vol20 to Vol60.
#
# At this stage, volatility is only a constructed candidate factor.
# Whether it contains information about future returns will be tested in D6.

vol20_f20_pearson = df["volatility_20_ann"].corr(df["future_return_20"])
vol20_f20_spearman = df["volatility_20_ann"].corr(df["future_return_20"], method="spearman")
print("Vol20-F20 Pearson =", vol20_f20_pearson)
print("Vol20-F20 Spearman =", vol20_f20_spearman)
vol_data = df[["volatility_20_ann", "future_return_20"]].dropna().copy()
vol_data["vol_group"] = pd.qcut(vol_data["volatility_20_ann"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
vol_group_return = vol_data.groupby("vol_group", observed=False)["future_return_20"].mean()
vol_group_count = vol_data.groupby("vol_group", observed=False)["future_return_20"].count()
print(vol_group_return)
print(vol_group_count)
vol_spread = vol_group_return["Q5"] - vol_group_return["Q1"]
print("Q5-Q1 spread =", vol_spread)
windows = [5, 20, 60]
horizons = [5, 20, 60]
vol_pearson_table = pd.DataFrame(index=windows, columns=horizons, dtype=float)
for window in windows:
    volatility = df["return"].rolling(window).std() * np.sqrt(252)
    for horizon in horizons:
        future_return = df["close"].pct_change(horizon).shift(-horizon)
        vol_pearson_table.loc[window, horizon] = volatility.corr(future_return)
print(vol_pearson_table)
vol_spearman_table = pd.DataFrame(index=windows, columns=horizons, dtype=float)
for window in windows:
    volatility = df["return"].rolling(window).std() * np.sqrt(252)
    for horizon in horizons:
        future_return = df["close"].pct_change(horizon).shift(-horizon)
        vol_spearman_table.loc[window, horizon] = volatility.corr(future_return, method="spearman")
print(vol_spearman_table)

df["future_return_60"] = df["close"].pct_change(60).shift(-60)
vol60_data = df[["volatility_20_ann", "future_return_60"]].dropna().copy()
vol60_data["vol_group"] = pd.qcut(vol60_data["volatility_20_ann"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
vol60_group_return = vol60_data.groupby("vol_group", observed=False)["future_return_60"].mean()
vol60_group_count = vol60_data.groupby("vol_group", observed=False)["future_return_60"].count()
print(vol60_group_return)
print(vol60_group_count)
print("Q5-Q1 spread =", vol60_group_return["Q5"] - vol60_group_return["Q1"])
df["year"] = df["date"].dt.year
for year in sorted(df["year"].unique()):
    pair = df[df["year"] == year][["volatility_20_ann", "future_return_60"]].dropna()
    r = pair["volatility_20_ann"].corr(pair["future_return_60"], method="spearman")
    print("year =", year, "n =", len(pair), "Spearman =", r)