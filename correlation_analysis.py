"""
感官评分与化学结构复杂度相关性分析
分析四个评分维度与三种结构复杂度之间的相关性
建立回归模型：用熟悉性、愉悦性、强度、化学复杂度解释感知复杂度
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. 加载数据 ====================
print("=" * 60)
print("1. 数据加载与预处理")
print("=" * 60)

data = pd.read_excel('data318.xlsx')
complexity = pd.read_excel('Complexity.xlsx')

# 混合物名称映射
mixture_map = {
    'IME': 'IME', 'NOP': 'NOP', 'CGA': 'CGA', 'OCP': 'OCP',
    'BCM': 'BCM', 'AMI': 'AMI', 'MIC': 'MIC', 'PCE': 'PCE', 'EGM': 'EGM'
}

# 提取每种混合物的评分均值
rating_cols = ['f', 'v', 'i', 'c']
mixture_ratings = {}

for mixture in mixture_map.keys():
    cols = [f'{mixture}-{r}' for r in rating_cols]
    if all(c in data.columns for c in cols):
        mixture_ratings[mixture] = {
            'f': data[cols[0]].mean(),
            'v': data[cols[1]].mean(),
            'i': data[cols[2]].mean(),
            'c': data[cols[3]].mean()
        }

ratings_df = pd.DataFrame(mixture_ratings).T
ratings_df.index.name = 'Mixture'
print("\n各混合物评分均值：")
print(ratings_df.round(3))

# 合并评分与复杂度数据
merged = ratings_df.reset_index().merge(complexity, on='Mixture')
print("\n合并后的数据：")
print(merged.round(3))

# ==================== 2. 相关性分析 ====================
print("\n" + "=" * 60)
print("2. 四个评分维度与结构复杂度的相关性分析")
print("=" * 60)

complexity_methods = ['Shannon', 'Average Vector', 'RdKit']
rating_dims = ['f', 'v', 'i', 'c']

# 创建相关性矩阵
corr_results = []

for rating_dim in rating_dims:
    for comp_method in complexity_methods:
        r, p = stats.pearsonr(merged[rating_dim], merged[comp_method])
        corr_results.append({
            '评分维度': rating_dim,
            '复杂度方法': comp_method,
            'Pearson r': round(r, 4),
            'p-value': round(p, 4),
            '显著性': '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        })

corr_df = pd.DataFrame(corr_results)
print("\n相关性分析结果：")
print(corr_df.to_string(index=False))

# 计算评分维度之间的相关矩阵
print("\n\n四个评分维度之间的相关性：")
rating_corr = ratings_df[['f', 'v', 'i', 'c']].corr()
print(rating_corr.round(3))

# ==================== 3. 多元线性回归（纯Python实现）====================
print("\n" + "=" * 60)
print("3. 多元线性回归模型")
print("=" * 60)
print("目标：用熟悉性(f)、愉悦性(v)、强度(i)、化学结构复杂度解释感知复杂度(c)")
print("-" * 60)

def linear_regression(X, y):
    """多元线性回归，使用最小二乘法"""
    n = len(y)
    # 添加常数项
    X_design = np.column_stack([np.ones(n), X])
    
    # 最小二乘解: β = (X'X)^(-1) X'y
    XtX = X_design.T @ X_design
    Xty = X_design.T @ y
    
    try:
        beta = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        # 奇异矩阵，使用伪逆
        beta = np.linalg.lstsq(X_design, y, rcond=None)[0]
    
    # 预测值
    y_pred = X_design @ beta
    
    # 残差
    residuals = y - y_pred
    
    # R²
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot
    
    # 调整R²
    p = X.shape[1]  # 特征数
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    # 标准误差
    se = np.sqrt(ss_res / (n - p - 1))
    
    # 系数标准误和t检验
    try:
        mse = ss_res / (n - p - 1)
        var_coef = mse * np.linalg.inv(XtX).diagonal()
        se_coef = np.sqrt(np.abs(var_coef))
        t_stats = beta[1:] / se_coef[1:]
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p - 1))
    except:
        t_stats = np.zeros(len(beta)-1)
        p_values = np.ones(len(beta)-1)
    
    return {
        'intercept': beta[0],
        'coefficients': beta[1:],
        'r2': r2,
        'adj_r2': adj_r2,
        'se': se,
        't_stats': t_stats,
        'p_values': p_values,
        'y_pred': y_pred,
        'residuals': residuals
    }

# 方法1：分别用三种复杂度建模
for comp_method in complexity_methods:
    print(f"\n【使用 {comp_method} 复杂度】")
    
    X = merged[['f', 'v', 'i', comp_method]].values
    y = merged['c'].values
    
    result = linear_regression(X, y)
    
    print(f"  R² = {result['r2']:.4f}, 调整R² = {result['adj_r2']:.4f}")
    print(f"  标准误差 = {result['se']:.4f}")
    print(f"  回归方程：c = {result['intercept']:.3f} + {result['coefficients'][0]:.3f}*f + {result['coefficients'][1]:.3f}*v + {result['coefficients'][2]:.3f}*i + {result['coefficients'][3]:.3f}*{comp_method}")
    print(f"  系数显著性：")
    for i, name in enumerate(['f', 'v', 'i', comp_method]):
        sig = '***' if result['p_values'][i] < 0.001 else '**' if result['p_values'][i] < 0.01 else '*' if result['p_values'][i] < 0.05 else 'ns'
        print(f"    {name}: β = {result['coefficients'][i]:.3f}, p = {result['p_values'][i]:.4f} {sig}")

# ==================== 4. 综合回归模型（使用标准化系数） ====================
print("\n" + "=" * 60)
print("4. 标准化回归分析（比较各变量相对重要性）")
print("=" * 60)

# 使用三种复杂度的平均值作为综合复杂度
merged['Avg_Complexity'] = merged[['Shannon', 'Average Vector', 'RdKit']].mean(axis=1)

X = merged[['f', 'v', 'i', 'Avg_Complexity']].values
y = merged['c'].values

# 标准化
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_scaled = (X - X_mean) / X_std
y_mean = y.mean()
y_std = y.std()
y_scaled = (y - y_mean) / y_std

result_std = linear_regression(X_scaled, y_scaled)

print(f"\n使用平均复杂度 (Shannon, Average Vector, RdKit 的均值)")
print(f"R² = {result_std['r2']:.4f}")
print(f"\n标准化回归系数（β，表示相对重要性）：")
for i, name in enumerate(['f (熟悉性)', 'v (愉悦性)', 'i (强度)', 'Avg_Complexity (化学复杂度)']):
    print(f"  {name}: β = {result_std['coefficients'][i]:.3f}")

# ==================== 5. 简化模型（只用感知维度） ====================
print("\n" + "=" * 60)
print("5. 简化模型：仅用感官评分解释感知复杂度")
print("=" * 60)

X_simple = merged[['f', 'v', 'i']].values
result_simple = linear_regression(X_simple, y)

print(f"仅使用 f, v, i 预测 c:")
print(f"  R² = {result_simple['r2']:.4f}, 调整R² = {result_simple['adj_r2']:.4f}")
print(f"  系数：")
for i, name in enumerate(['f (熟悉性)', 'v (愉悦性)', 'i (强度)']):
    sig = '***' if result_simple['p_values'][i] < 0.001 else '**' if result_simple['p_values'][i] < 0.01 else '*' if result_simple['p_values'][i] < 0.05 else 'ns'
    print(f"    {name}: β = {result_simple['coefficients'][i]:.3f}, p = {result_simple['p_values'][i]:.4f} {sig}")

# ==================== 6. 结果汇总 ====================
print("\n" + "=" * 60)
print("6. 分析结论汇总")
print("=" * 60)

# 找出最显著的相關
print("\n【评分维度与复杂度的相关性】")
for comp in complexity_methods:
    strongest = corr_df[corr_df['复杂度方法'] == comp].sort_values('Pearson r', key=abs, ascending=False).iloc[0]
    print(f"  {comp}: {strongest['评分维度']}与r={strongest['Pearson r']} (p={strongest['p-value']})")

print("\n【回归模型预测能力】")
for comp in complexity_methods:
    X_temp = merged[['f', 'v', 'i', comp]].values
    result = linear_regression(X_temp, y)
    print(f"  {comp}: R² = {result['r2']:.4f}")

# 保存结果
results_summary = corr_df.copy()
results_summary.to_csv('correlation_results.csv', index=False, encoding='utf-8-sig')
print("\n相关性结果已保存至: correlation_results.csv")