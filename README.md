# Odorant Mixture Perceptual Similarity Analysis

基于 Snitz et al. (2013) PLoS Computational Biology 论文的气味混合物感知相似度预测分析工具。

**论文原文**: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003184

## 📁 文件说明

### 数据文件
| 文件 | 说明 |
|------|------|
| `13-descriptors-dragon.xlsx` | 13个分子的Dragon分子描述符（1666个描述符） |
| `mixture-components.xlsx` | 9个混合物的成分组成 |
| `file.pdf` | 参考论文 PDF |

### Python 脚本
| 文件 | 功能 |
|------|------|
| `molecular_similarity_heatmap.py` | 计算13个分子之间的余弦相似度并生成热图 |
| `mixture_similarity_heatmap.py` | 计算9个混合物之间的相似度，包含聚类分析 |
| `mixture_analysis.py` | 综合分析：热图 + 聚类 + 复杂度柱状图（publication版本） |
| `mixture_similarity_final.py` | 最终版本，支持自动分析所有混合物 |

### 输出结果
| 文件 | 说明 |
|------|------|
| `similarity_heatmap.png` | 13个分子的相似度热图 |
| `molecular_similarity_matrix.xlsx` | 分子相似度矩阵数据 |
| `mixture_similarity_heatmap.png` | 混合物相似度热图（带聚类树） |
| `mixture_clustering_pub.png` | 论文级聚类图 |
| `mixture_complexity_bar.png` | 复杂度柱状图 |
| `mixture_similarity_matrix.xlsx` | 混合物相似度矩阵数据 |

---

## 🧪 方法论

### 1. 描述符归一化
将每个 Dragon 描述符归一化到 [0,1] 范围：
```
vn = (v - min(ld)) / (max(ld) - min(ld))
```

### 2. 混合物向量
混合物总向量 = 各组分描述符向量之和，然后归一化：
```
MixVector = Σ(component descriptors) / ||Σ(component descriptors)||
```

### 3. 余弦相似度
```
cosine_similarity = (A · B) / (||A|| × ||B||)
```

### 4. 信息熵与复杂度
```
H = -Σ(p × log2(p))
RPC = H / log2(n_descriptors)
```

---

## 🚀 使用方法

### 环境配置

```bash
# 创建虚拟环境
cd ~/workspace/1
python3 -m venv venv2
source venv2/bin/activate

# 安装依赖
pip install pandas numpy openpyxl scipy seaborn matplotlib
```

### 运行分析

#### 1. 分子相似度热图
```bash
python molecular_similarity_heatmap.py
# 输出: similarity_heatmap.png, molecular_similarity_matrix.xlsx
```

#### 2. 混合物相似度热图 + 聚类
```bash
python mixture_similarity_heatmap.py
# 输出: mixture_similarity_heatmap.png, mixture_similarity_matrix.xlsx
```

#### 3. 综合分析（论文级图表）
```bash
python mixture_analysis.py
# 输出:
#   - mixture_similarity_heatmap_pub.png  (热图)
#   - mixture_clustering_pub.png          (聚类图)
#   - mixture_complexity_bar.png         (复杂度柱状图)
```

---

## 📊 分析结果

### 13个分子（余弦相似度最高5对）
| 分子对 | 相似度 |
|--------|--------|
| 2-Nonanol - 2-Octanone | 0.92 |
| Acetophenone - Guaiacol | 0.92 |
| Eugenol - Methyl anthranilate | 0.91 |
| Pentyl valerate - 2-Nonanol | 0.91 |
| Guaiacol - Methyl anthranilate | 0.90 |

### 9个混合物（层次聚类）
- **簇1**: PCE, OCP（含 Cinnamaldehyde/Eugenol/Pentyl valerate）
- **簇2**: MIC, IME（含 5-Methylfurfural/Carvone）
- **簇3**: EGM, CGA, AMI, BCM（含 Acetophenone/Guaiacol）

### 混合物复杂度排名
| 复杂度 | 混合物 |
|--------|--------|
| 0.953 | IME |
| 0.952 | OCP |
| 0.951 | PCE |
| 0.951 | NOP |
| 0.947 | BCM |
| 0.947 | MIC |
| 0.944 | AMI |
| 0.943 | CGA |
| 0.934 | EGM |

---

## 📝 引用

如果使用本代码，请引用：

> Snitz, K., et al. (2013) Predicting Odor Perceptual Similarity from Odor Structure. PLoS Comput Biol 9(9): e1003184

---

## 📄 License

MIT License
