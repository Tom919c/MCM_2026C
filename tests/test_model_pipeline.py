"""
测试贝叶斯模型是否能正常运行
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("Step 1: Importing...")
sys.stdout.flush()

from src.config_loader import load_config
from src.bayesian_model.data_utils import load_data, validate_data, filter_data
from src.bayesian_model.model import build_model, train

print("Step 2: Loading config and data...")
sys.stdout.flush()

config = load_config('configs/config.yaml')
datas = {}
datas = load_data('data/processed/datas.pkl', datas)

print("Step 3: Filtering data...")
sys.stdout.flush()

train_datas, test_datas = filter_data(datas, 0)
print(f"  train: {train_datas['train_data']['n_obs']} obs")

print("Step 4: Building model...")
sys.stdout.flush()

model_params = build_model(config, train_datas)
print(f"  Model params ready: {list(model_params.keys())}")

print("Step 5: Testing MCMC (10 samples)...")
sys.stdout.flush()

# 临时修改采样配置为快速测试
config['sampling']['n_tune'] = 10
config['sampling']['n_samples'] = 10
config['sampling']['n_chains'] = 1

train_datas = train(config, model_params, train_datas)
print("  MCMC completed!")

print("Step 6: Checking results...")
sys.stdout.flush()
samples = train_datas['trace']
print(f"  Sample keys: {list(samples.keys())}")
print(f"  alpha shape: {samples['alpha'].shape}")

print("All tests passed!")
