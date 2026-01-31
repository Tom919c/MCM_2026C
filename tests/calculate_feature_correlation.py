"""
计算训练特征之间的相关系数矩阵

输出CSV文件包含所有特征两两之间的相关系数
"""

import pandas as pd
import pickle
import numpy as np
from pathlib import Path


def load_train_data(data_path):
    """加载训练数据并构建完整特征矩阵"""
    print("正在加载训练数据...")
    with open(data_path, 'rb') as f:
        datas = pickle.load(f)

    if 'train_data' not in datas:
        raise KeyError("数据字典中未找到 'train_data' 键")

    train_data = datas['train_data']

    print(f"观测数量: {train_data['n_obs']}")
    print(f"选手特征数: {len(train_data['X_celeb_names'])}")
    print(f"职业舞者特征数: {len(train_data['X_pro_names'])}")
    print(f"观测特征数: {len(train_data['X_obs_names'])}")

    # 构建完整特征矩阵
    print("\n正在构建完整特征矩阵...")

    # 1. 根据索引扩展选手特征
    celeb_idx = train_data['celeb_idx']
    X_celeb_expanded = train_data['X_celeb'][celeb_idx]  # (n_obs, n_celeb_features)

    # 2. 根据索引扩展职业舞者特征
    pro_idx = train_data['pro_idx']
    X_pro_expanded = train_data['X_pro'][pro_idx]  # (n_obs, n_pro_features)

    # 3. 观测特征已经是 (n_obs, n_obs_features)
    X_obs = train_data['X_obs']

    # 4. 合并所有特征
    X_all = np.concatenate([X_celeb_expanded, X_pro_expanded, X_obs], axis=1)

    # 5. 构建特征名称列表
    feature_names = (
        train_data['X_celeb_names'] +
        ['pro_' + name for name in train_data['X_pro_names']] +
        train_data['X_obs_names']
    )

    # 6. 转换为DataFrame
    train_df = pd.DataFrame(X_all, columns=feature_names)

    print(f"完整特征矩阵形状: {train_df.shape}")
    print(f"总特征数: {len(feature_names)}")

    return train_df


def identify_feature_columns(train_df):
    """识别特征列（所有列都是特征）"""
    print("\n正在识别特征列...")

    feature_cols = train_df.columns.tolist()

    print(f"特征列数量: {len(feature_cols)}")
    print("\n特征列表:")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i}. {col}")

    return feature_cols


def calculate_correlation_matrix(train_df, feature_cols):
    """计算特征之间的相关系数矩阵"""
    print("\n正在计算相关系数矩阵...")

    # 提取特征数据
    feature_data = train_df[feature_cols].copy()

    # 处理缺失值
    missing_count = feature_data.isnull().sum().sum()
    if missing_count > 0:
        print(f"警告: 发现 {missing_count} 个缺失值，将使用均值填充")
        feature_data = feature_data.fillna(feature_data.mean())

    # 计算相关系数矩阵（Pearson相关系数）
    corr_matrix = feature_data.corr()

    print(f"相关系数矩阵形状: {corr_matrix.shape}")

    return corr_matrix


def analyze_correlation(corr_matrix, threshold=0.8):
    """分析高相关性特征对"""
    print(f"\n正在分析高相关性特征对（阈值: {threshold}）...")

    # 获取上三角矩阵（避免重复）
    upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    corr_upper = corr_matrix.where(upper_triangle)

    # 找出高相关性的特征对
    high_corr_pairs = []

    for col in corr_upper.columns:
        for idx in corr_upper.index:
            corr_value = corr_upper.loc[idx, col]
            if pd.notna(corr_value) and abs(corr_value) >= threshold:
                high_corr_pairs.append({
                    'feature_1': idx,
                    'feature_2': col,
                    'correlation': corr_value
                })

    # 按相关系数绝对值排序
    high_corr_pairs = sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)

    print(f"发现 {len(high_corr_pairs)} 对高相关性特征")

    if high_corr_pairs:
        print("\n前10对高相关性特征:")
        for i, pair in enumerate(high_corr_pairs[:10], 1):
            print(f"  {i}. {pair['feature_1']} <-> {pair['feature_2']}: {pair['correlation']:.4f}")

    return high_corr_pairs


def save_correlation_matrix(corr_matrix, output_path):
    """保存相关系数矩阵到CSV文件"""
    print(f"\n正在保存相关系数矩阵到: {output_path}")

    # 保存为CSV
    corr_matrix.to_csv(output_path, encoding='utf-8-sig')

    print("保存完成")


def main():
    """主函数"""
    print("=" * 60)
    print("训练特征相关系数矩阵计算")
    print("=" * 60)

    # 配置路径
    data_path = 'data/processed/datas.pkl'
    output_path = 'outputs/feature_correlation_matrix.csv'

    # 创建输出目录
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # 1. 加载训练数据
    train_df = load_train_data(data_path)

    # 2. 识别特征列
    feature_cols = identify_feature_columns(train_df)

    if len(feature_cols) == 0:
        print("错误: 未找到任何特征列")
        return

    # 3. 计算相关系数矩阵
    corr_matrix = calculate_correlation_matrix(train_df, feature_cols)

    # 4. 分析高相关性特征对
    high_corr_pairs = analyze_correlation(corr_matrix, threshold=0.8)

    # 5. 保存相关系数矩阵
    save_correlation_matrix(corr_matrix, output_path)

    # 6. 保存高相关性特征对
    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs)
        high_corr_output = 'outputs/high_correlation_pairs.csv'
        high_corr_df.to_csv(high_corr_output, index=False, encoding='utf-8-sig')
        print(f"高相关性特征对已保存到: {high_corr_output}")

    print("\n" + "=" * 60)
    print("计算完成")
    print("=" * 60)


if __name__ == '__main__':
    main()

