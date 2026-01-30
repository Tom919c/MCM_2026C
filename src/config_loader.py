"""
配置加载器模块

提供配置文件加载功能
"""

import yaml
from pathlib import Path


def load_config(config_path):
    """
    加载YAML配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        config: 配置字典
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config
