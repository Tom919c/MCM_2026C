# 问题1分析可视化说明

## 概述

本模块实现了问题1的所有可视化需求，基于贝叶斯模型的输出生成分析图表。

## 可视化清单

### 6.1 粉丝投票估算结果呈现

#### 6.1.1 选手每周粉丝投票趋势
- **文件**: `contestant_trend_{celeb_idx}_{name}.png`
- **内容**:
  - 上图：P_fan随周次变化，带95%后验置信区间
  - 下图：潜在强度μ随周次变化
- **示例选手**: Bobby Bones (S27冠军), Jerry Rice (S2亚军)

#### 6.1.2 赛季粉丝投票分布
- **文件**:
  - `pfan_stacked_area_{season_id}.png` - 堆叠面积图
  - `pfan_heatmap_{season_id}.png` - 热力图
- **内容**: 展示整个赛季每周所有选手的P_fan分布

#### 6.1.3 淘汰预测准确率
- **文件**:
  - `elimination_accuracy.png` - 柱状图
  - `elimination_accuracy_summary.csv` - 统计表格
- **内容**: 每周淘汰预测准确率及平均值

#### 6.1.4 存活周数分布对比
- **文件**: `survival_distribution_comparison.png`
- **内容**:
  - 左图：直方图对比模拟vs真实
  - 右图：KDE曲线对比

#### 6.1.5 样条函数曲线
- **文件**: `spline_{feature_name}.png`
- **内容**: 非线性特征（z_score, weeks_survived等）对μ的影响曲线，带95%置信区间

### 6.2 关键特征影响分析

#### 6.2.1 特征重要性Top 10
- **文件**:
  - `feature_importance_top10.png` - 条形图
  - `feature_importance_top10.csv` - 详细数据
- **内容**: 按|β|降序排列的Top 10特征，带95%可信区间误差棒

### 6.3 估算确定性评估

#### 6.3.1 交叉验证准确率
- **文件**:
  - `cv_accuracy_trend.png` - 柱状图
  - `cv_accuracy_summary.csv` - 统计表格
- **内容**: 留一赛季法交叉验证结果，每个赛季的预测准确率

#### 6.3.2 参数后验分布
- **文件**: `posterior_distributions.png`
- **内容**: 关键参数（τ, ν, θ_save等）的后验分布直方图和KDE

## 使用方法

### 1. 确保已完成模型训练

```bash
activate MCM
python pipelines/train_pipeline.py
```

训练完成后会生成 `data/models/training_results.pkl`

### 2. 运行可视化流程

```bash
activate MCM
python pipelines/analysis_visualization_pipeline.py
```

### 3. 查看结果

所有图表保存在 `outputs/analysis/` 目录下

## 数据要求

可视化流程需要以下数据（来自 `training_results.pkl`）：

1. **model_output**: 模型输出字典
   - mu, P_fan: 预测结果
   - alpha_contrib, delta_contrib, linear_contrib, spline_contrib: 贡献分解
   - alpha, delta, beta_obs: 模型参数
   - feature_names: 特征名称

2. **train_data**: 训练数据字典
   - celeb_idx, pro_idx, week_idx, season_idx: 索引数组
   - X_celeb, X_pro, X_obs: 特征矩阵
   - week_data: 周级结构数据

3. **posterior_samples** (可选): 后验样本字典
   - 用于计算置信区间
   - 如果没有，将使用点估计

4. **metrics** (可选): 评估指标字典
   - week_results: 每周预测结果
   - mean_accuracy: 平均准确率

5. **cv_results** (可选): 交叉验证结果
   - 用于绘制CV准确率趋势

## 自定义选项

### 修改代表性选手

编辑 `pipelines/analysis_visualization_pipeline.py` 中的 `contestant_names`:

```python
contestant_names = [
    {'celeb_idx': 5, 'name': 'Bobby Bones'},
    {'celeb_idx': 12, 'name': 'Jerry Rice'},
    # 添加更多选手
]
```

### 修改Top N特征数量

```python
plot_feature_importance(model_output, posterior_samples, top_n=15, output_dir=output_dir)
```

### 选择特定赛季

```python
plot_season_pfan_distribution(model_output, train_data, season_id=2, output_dir=output_dir)
```

## 技术细节

### 置信区间计算

- 使用后验样本的2.5%和97.5%分位数
- 如果没有后验样本，使用点估计（无置信区间）

### 样条函数可视化

- 从 `model_output['feature_names']['splines']` 获取样条特征列表
- 对每个特征绘制其对μ的贡献曲线

### 交叉验证结果

- 需要在训练时运行 `run_cv()` 函数
- 结果保存在 `datas['cv_results']`

## 依赖包

```bash
pip install numpy pandas matplotlib seaborn scipy
```

## 注意事项

1. **中文字体**: 如果图表中文显示为方框，需要安装SimHei字体或修改 `plt.rcParams['font.sans-serif']`

2. **内存占用**: 如果后验样本很大，计算置信区间可能占用较多内存

3. **文件命名**: 选手名称中的空格会被替换为下划线

4. **数据完整性**: 某些可视化需要特定数据，如果缺失会跳过并显示警告

## 输出示例

```
outputs/analysis/
├── contestant_trend_0_Contestant_0.png
├── contestant_trend_1_Contestant_1.png
├── pfan_stacked_area_season_0.png
├── pfan_heatmap_season_0.png
├── pfan_stacked_area_all_seasons.png
├── pfan_heatmap_all_seasons.png
├── elimination_accuracy.png
├── elimination_accuracy_summary.csv
├── survival_distribution_comparison.png
├── spline_z_score.png
├── spline_weeks_survived.png
├── feature_importance_top10.png
├── feature_importance_top10.csv
├── cv_accuracy_trend.png
├── cv_accuracy_summary.csv
└── posterior_distributions.png
```
