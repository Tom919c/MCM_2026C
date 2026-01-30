"""
数据管理器模块

提供数据加载、保存功能
"""

import pandas as pd
import pickle
from pathlib import Path


def read_csv_with_encoding(file_path):
    """
    尝试多种编码读取CSV文件

    Args:
        file_path: CSV文件路径

    Returns:
        df: pandas DataFrame
    """
    encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252', 'gbk']

    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue

    # 最后使用utf-8并忽略错误
    return pd.read_csv(file_path, encoding='utf-8', errors='ignore')


def load_raw_data(config, datas):
    """
    加载原始CSV数据

    Args:
        config: 配置字典
        datas: 数据字典

    Returns:
        datas: 更新后的数据字典，添加 'raw_data' 键
    """
    raw_data_path = config.get('raw_data_path', 'data/raw/2026_MCM_Problem_C_Data.csv')

    print(f"正在加载原始数据: {raw_data_path}")
    df = read_csv_with_encoding(raw_data_path)
    print(f"数据加载完成，共 {len(df)} 行，{len(df.columns)} 列")

    datas['raw_data'] = df
    return datas


def save_data(datas, file_path):
    """
    保存数据字典到文件

    Args:
        datas: 数据字典
        file_path: 保存路径（.pkl文件）
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'wb') as f:
        pickle.dump(datas, f)

    print(f"数据已保存到: {file_path}")


def load_data(file_path):
    """
    从文件加载数据字典

    Args:
        file_path: 数据文件路径（.pkl文件）

    Returns:
        datas: 数据字典
    """
    with open(file_path, 'rb') as f:
        datas = pickle.load(f)

    print(f"数据已从 {file_path} 加载")
    return datas
