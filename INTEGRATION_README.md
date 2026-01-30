# 贝叶斯模型训练代码整合说明

## 整合完成

队友的模型训练代码已成功整合到项目中。

## 目录结构

```
MCM_2026C/
├── configs/
│   └── config.yaml                    # 合并后的配置文件（包含模型训练配置）
├── src/
│   └── bayesian_model/                # 新增：贝叶斯模型模块
│       ├── __init__.py
│       ├── model.py                   # NumPyro 贝叶斯模型
│       └── data_utils.py              # 数据加载和验证工具
└── pipelines/
    ├── train_pipeline.py              # 模型训练主流程
    └── test_model_pipeline.py         # 模型测试脚本
```

## 主要修改

### 1. 文件移动和重命名
- `train—\C-bayes\src\model.py` → `src/bayesian_model/model.py`
- `train—\C-bayes\src\preprocess.py` → `src/bayesian_model/data_utils.py`
- `train—\C-bayes\main.py` → `pipelines/train_pipeline.py`
- `train—\C-bayes\test_model.py` → `pipelines/test_model_pipeline.py`

### 2. 路径更新
- 配置文件路径：`config.yaml` → `configs/config.yaml`
- 数据文件路径：`datas.pkl` → `data/processed/datas.pkl`
- 输出文件路径：`results.pkl` → `data/models/training_results.pkl`
- 输出目录：`outputs/model` → `outputs/training`

### 3. 导入路径更新
```python
# 原来
from src.utils import load_config, save_data
from src.preprocess import load_data, validate_data
from src.model import build_model, train

# 现在
from src.config_loader import load_config
from src.data_manager import save_data
from src.bayesian_model.data_utils import load_data, validate_data
from src.bayesian_model.model import build_model, train
```

### 4. 配置文件合并
将队友的 `config.yaml` 合并到 `configs/config.yaml`，新增以下配置项：
- `model`: 模型结构配置（样条特征、正则化等）
- `prior`: 先验分布参数
- `sampling`: MCMC 采样配置
- `training_output`: 训练输出配置

## 使用方法

### 1. 测试模型（快速验证）
```bash
activate MCM
python pipelines/test_model_pipeline.py
```

### 2. 完整训练流程
```bash
activate MCM
python pipelines/train_pipeline.py
```

训练流程包括：
1. 加载数据（`data/processed/datas.pkl`）
2. 数据校验
3. 留一赛季交叉验证（Leave-One-Season-Out CV）
4. 全量数据训练
5. 保存结果到 `data/models/training_results.pkl`

## 输出文件

训练完成后会生成：
- `data/models/training_results.pkl`: 包含后验样本、模型输出、评估指标
- `outputs/training/`: 训练过程的可视化结果（如果配置了保存）

## 依赖包

确保已安装以下包：
```bash
pip install numpyro jax jaxlib patsy tqdm
```

## 注意事项

1. **数据格式**：确保 `data/processed/datas.pkl` 包含符合 `数据接口规范.md` 的 `train_data`
2. **计算资源**：MCMC 采样需要较长时间，建议在性能较好的机器上运行
3. **配置调整**：可以在 `configs/config.yaml` 中调整采样参数（链数、样本数等）

## 原始代码位置

队友的原始代码保留在：`D:\Code\VSCode\MCM_2026C\train—\C-bayes\`
