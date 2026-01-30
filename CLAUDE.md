## Claude’s Role

Claude 在本项目中应扮演：

- 一名**美国大学生数学建模竞赛专家**
- 熟悉统计建模、概率论、机器学习
- 能清楚解释建模假设、方法选择的动机与局限
- 具有MCM竞赛风格的问题分析能力

## 编程要求

- 只有代码注释、claude回答、命令行打印信息用中文，其它地方一律用英文
- 图表上的标注，变量、函数等的命名用英文
- 命令行输出不要有emoji符号，只能用文本
- 在处理大规模计算时，优先考虑使用pandas和numpy进行向量化运算，而不是用python循环
- 在编写代码时，你的问题解决方案必须全都都由用户来提供或者你提供解决思路用户确认后你才可编写代码，如果遇到因为用户疏忽没有给你某个问题的解决思路，你必须立刻停止编写代码并询问用户问题如何解决
- 不可自行运行代码，得到用户让你运行代码的指令后才可运行

## Python Environment

- 使用 conda 环境
- 环境名称：MCM
- 所有代码均默认在 MCM 环境下运行
- 激活命令： activate MCM


## 编码问题处理规范

**重要：在处理文件读写时，必须考虑编码兼容性**

在读取CSV文件时，**不要直接使用单一编码**（如`encoding='utf-8'`），而应该：

1. **使用多编码尝试机制**：
   
   ```python
   def read_csv_with_encoding(file_path):
       """尝试多种编码读取CSV文件"""
       encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252', 'gbk']
       for encoding in encodings:
           try:
               return pd.read_csv(file_path, encoding=encoding)
           except UnicodeDecodeError:
               continue
       # 最后使用utf-8并忽略错误
       return pd.read_csv(file_path, encoding='utf-8', errors='ignore')
   
2. 常见编码顺序：
  - utf-8-sig：处理带BOM的UTF-8文件
  - utf-8：标准UTF-8
  - latin-1 / cp1252：Windows常用编码
  - gbk：中文Windows系统

在写入文件时：使用encoding='utf-8-sig'确保跨平台兼容

原则

- 永远不要假设文件编码
- 提供编码容错机制
- 在Windows环境下尤其注意编码问题


# MCM C题通用代码结构设计

## 1. 配置加载器

- 功能：加载YAML配置文件，返回字典
- 接口：`config = load_config(config_path)`

## 2. 数据字典

- 结构：普通Python字典，存放所有数据，将所有处理的数据均存入一个全局数据字典统一管理
- 辅助函数：
  - `save_data(data_dict, path)` - 保存数据字典到文件
  - `data_dict = load_data(path)` - 从文件加载数据字典

## 3. 统一数据流设计

- 核心：全局数据字典 `datas`
- 所有操作都是链式更新 `datas`
- 数据处理：`datas = process_xxx(config, datas)`
- 模型训练：`model = train(config, model, datas)`
- 模型生成：`datas = generate_xxx(config, model, datas)`
- 特点：
  - 统一的函数签名
  - 链式调用
  - 数据字典不断累积和更新

## 4. 模型加载器

- 功能：根据配置加载模型
- 接口：`model = load_model(config)`

## 5. 训练模块

- 功能：训练模型
- 接口：`model = train(config, model, datas)`

## 6. 模型生成数据模块

- 功能：使用模型进行推理/预测，生成数据
- 接口：`datas = generate_xxx(config, model, datas)`

## 7. 可视化模块

- 功能：生成图表并保存到文件
- 接口：`visualize_xxx(config, datas, tag)`
- 输出路径：`outputs/{tag}/`
- 特点：
  - tag硬编码在调用时指定
  - 纯副作用函数，不返回值
  - 只保存图表到对应目录

## 8. 目录结构
```


```

## 9. 函数式方法示例

### 示例1：数据处理流程

```python
# 加载配置
config = load_config('config.yaml')

# 初始化数据字典
datas = {}

# 数据处理链式调用
datas = load_raw_data(config, datas)           # 加载原始数据
datas = clean_data(config, datas)              # 数据清洗
datas = feature_engineering(config, datas)     # 特征工程
datas = split_dataset(config, datas)           # 划分数据集

# 保存处理后的数据
save_data(datas, 'processed_data.pkl')
```

### 示例2：完整建模流程

```python
# 加载配置和数据
config = load_config('config.yaml')
datas = load_data('processed_data.pkl')

# 加载模型
model = load_model(config)

# 训练模型
model = train(config, model, datas)

# 模型生成数据
datas = predict(config, model, datas)          # 生成预测结果
datas = evaluate_metrics(config, model, datas) # 计算评估指标
datas = analyze_results(config, model, datas)  # 结果分析

# 可视化
visualize_predictions(config, datas, 'predictions')
visualize_metrics(config, datas, 'metrics')
visualize_analysis(config, datas, 'analysis')

# 保存最终结果
save_data(datas, 'final_results.pkl')
```

## 10. 项目实战示例

### 示例1：训练流程 (train_pipeline.py)

```python
"""
清晰的函数式流程，每一步都明确输入输出
"""

# 加载配置
config = load_config('config.yaml')

# 初始化数据字典
datas = {}

# ========== 数据准备阶段 ==========
datas = load_raw_data(config, datas)              # 加载原始CSV数据
datas = preprocess_xxx(config, datas)        	  # 数据预处理
datas = build_features(config, datas)             # 特征工程
datas = split_train_test(config, datas)           # 划分训练/测试集

# 保存预处理数据
save_data(datas, 'processed_data.pkl')

# ========== 模型训练阶段 ==========
model = load_model(config)                        # 加载模型架构
model = train(config, model, datas)               # 训练模型
save_model(model, 'trained_model.pth')            # 保存模型

# ========== 评估阶段 ==========
datas = predict_train_set(config, model, datas)   # 训练集预测
datas = predict_test_set(config, model, datas)    # 测试集预测
datas = calculate_metrics(config, model, datas)   # 计算评估指标

# ========== 可视化阶段 ==========
visualize_training_history(config, datas, 'training')
visualize_predictions(config, datas, 'predictions')
visualize_metrics(config, datas, 'metrics')

# 保存最终结果
save_data(datas, 'train_results.pkl')

print("训练流程完成！")
```

### 示例2：预测流程 (predict_pipeline.py)

```python
"""
专注于预测任务，流程简洁明了
"""

# 加载配置和已训练模型
config = load_config('config.yaml')
model = load_model_weights('trained_model.pth')

# 加载预处理数据
datas = load_data('processed_data.pkl')

# ========== 预测 ==========
datas = prepare_features(config, datas)           # 准备特征
datas = predict_2028_gold(config, model, datas)   # 预测金牌
datas = predict_2028_medals(config, model, datas) # 预测总奖牌
datas = rank_countries(config, datas)             # 国家排名
datas = analyze_top_countries(config, datas)      # Top国家分析

# ========== 可视化预测结果 ==========
visualize_medal_distribution(config, datas, '2028_distribution')
visualize_country_ranking(config, datas, '2028_ranking')
visualize_sport_breakdown(config, datas, '2028_sports')

# 保存预测结果
save_data(datas, 'predictions_2028.pkl')
export_to_csv(datas, 'predictions_2028.csv')

print("2028预测完成！")
```

### 示例3：分析流程 (analysis_pipeline.py)

```python
"""
多维度分析，流程清晰可扩展
"""

# 加载配置和数据
config = load_config('config.yaml')
datas = load_data('train_results.pkl')

# ========== 稳定性分析 ==========
datas = calculate_country_stability(config, datas)
datas = identify_volatile_countries(config, datas)
datas = analyze_stability_factors(config, datas)
visualize_stability(config, datas, 'stability_analysis')

# ========== 策略分析 ==========
datas = analyze_medal_strategies(config, datas)
datas = identify_specialist_countries(config, datas)
datas = compare_strategies(config, datas)
visualize_strategies(config, datas, 'strategy_analysis')

# 保存分析结果
save_data(datas, 'analysis_results.pkl')

print("深度分析完成！")
```

### 示例4：快速实验流程 (experiment_pipeline.py)

```python
"""
快速实验 - 测试不同配置
展示如何轻松切换配置进行实验
"""

# 实验1：基础模型
config1 = load_config('config_baseline.yaml')
datas1 = load_data('processed_data.pkl')
model1 = load_model(config1)
model1 = train(config1, model1, datas1)
datas1 = evaluate(config1, model1, datas1)
save_data(datas1, 'exp1_baseline.pkl')

# 实验2：添加运动员强度特征
config2 = load_config('config_with_strength.yaml')
datas2 = load_data('processed_data.pkl')
datas2 = add_athlete_strength(config2, datas2)
model2 = load_model(config2)
model2 = train(config2, model2, datas2)
datas2 = evaluate(config2, model2, datas2)
save_data(datas2, 'exp2_strength.pkl')

# 实验3：不同损失函数
config3 = load_config('config_focal_loss.yaml')
datas3 = load_data('processed_data.pkl')
model3 = load_model(config3)
model3 = train(config3, model3, datas3)
datas3 = evaluate(config3, model3, datas3)
save_data(datas3, 'exp3_focal.pkl')

# 对比实验结果
datas_compare = {}
datas_compare = load_experiment(config1, datas_compare, 'exp1_baseline.pkl', 'baseline')
datas_compare = load_experiment(config2, datas_compare, 'exp2_strength.pkl', 'strength')
datas_compare = load_experiment(config3, datas_compare, 'exp3_focal.pkl', 'focal')
datas_compare = compare_experiments(config1, datas_compare)

visualize_experiment_comparison(config1, datas_compare, 'experiment_comparison')

print("实验对比完成！")
```

## 11. 函数式方法的优势

**核心优势：**

1. **流程清晰**：每一步都是独立的函数调用，一目了然
2. **易于调试**：可以单独测试每个函数，快速定位问题
3. **易于扩展**：添加新步骤只需插入一行代码
4. **易于实验**：可以轻松注释/取消注释某些步骤
5. **易于并行**：不同实验可以独立运行，互不干扰
6. **统一接口**：所有函数都遵循相同的签名模式
7. **数据可追溯**：datas字典记录了所有中间结果