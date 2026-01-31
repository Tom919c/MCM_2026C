"""
Test YouTube Features Addition

测试YouTube特征添加功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
from src.youtube_features import add_youtube_features_to_train_data


def load_datas(data_path):
    """加载已保存的datas字典"""
    print(f"加载datas字典: {data_path}")
    with open(data_path, 'rb') as f:
        datas = pickle.load(f)
    return datas


def print_youtube_feature_stats(datas):
    """打印YouTube特征统计信息"""
    train_data = datas['train_data']
    X_obs = train_data['X_obs']
    X_obs_names = train_data['X_obs_names']

    # 找到YouTube特征的索引
    youtube_feature_indices = {}
    for feat_name in ['has_youtube_video', 'youtube_view_count_norm',
                      'youtube_like_count_norm', 'youtube_comment_count_norm']:
        if feat_name in X_obs_names:
            youtube_feature_indices[feat_name] = X_obs_names.index(feat_name)

    if not youtube_feature_indices:
        print("未找到YouTube特征")
        return

    print("\n" + "=" * 60)
    print("YouTube特征统计")
    print("=" * 60)

    # 统计has_youtube_video
    has_video_idx = youtube_feature_indices['has_youtube_video']
    has_video = X_obs[:, has_video_idx]
    n_with_video = np.sum(has_video == 1.0)
    n_total = len(has_video)

    print(f"\n1. has_youtube_video:")
    print(f"   - 有视频数据的观测: {n_with_video} / {n_total} ({n_with_video / n_total * 100:.2f}%)")
    print(f"   - 无视频数据的观测: {n_total - n_with_video} / {n_total} ({(n_total - n_with_video) / n_total * 100:.2f}%)")

    # 统计归一化特征（只统计有视频的观测）
    for feat_name in ['youtube_view_count_norm', 'youtube_like_count_norm', 'youtube_comment_count_norm']:
        feat_idx = youtube_feature_indices[feat_name]
        feat_values = X_obs[:, feat_idx]

        # 只统计有视频的观测
        feat_values_with_video = feat_values[has_video == 1.0]

        print(f"\n2. {feat_name}:")
        print(f"   - 均值: {feat_values_with_video.mean():.4f}")
        print(f"   - 标准差: {feat_values_with_video.std():.4f}")
        print(f"   - 最小值: {feat_values_with_video.min():.4f}")
        print(f"   - 最大值: {feat_values_with_video.max():.4f}")
        print(f"   - 中位数: {np.median(feat_values_with_video):.4f}")

    # 按赛季统计覆盖率
    print("\n" + "=" * 60)
    print("按赛季统计YouTube数据覆盖率")
    print("=" * 60)

    season_idx = train_data['season_idx']
    unique_seasons = np.unique(season_idx)

    print(f"\n{'Season':<10} {'Total Obs':<12} {'With Video':<12} {'Coverage':<12}")
    print("-" * 60)

    for season in sorted(unique_seasons):
        season_mask = (season_idx == season)
        n_season_obs = np.sum(season_mask)
        n_season_with_video = np.sum((season_mask) & (has_video == 1.0))
        coverage = n_season_with_video / n_season_obs * 100 if n_season_obs > 0 else 0

        print(f"{int(season):<10} {n_season_obs:<12} {int(n_season_with_video):<12} {coverage:.2f}%")

    print("=" * 60)


def main():
    """主函数"""
    # 定义文件路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, 'data', 'processed', 'datas.pkl')

    # 检查文件是否存在
    if not os.path.exists(data_path):
        print(f"错误: datas文件不存在: {data_path}")
        print("请先运行训练数据构建流程")
        return

    # 加载datas字典
    datas = load_datas(data_path)

    print(f"\n原始X_obs shape: {datas['train_data']['X_obs'].shape}")
    print(f"原始X_obs_names: {len(datas['train_data']['X_obs_names'])} 个特征")

    # 添加YouTube特征
    config = {}  # 空配置，因为函数内部会自动查找文件路径
    datas = add_youtube_features_to_train_data(config, datas)

    print(f"\n更新后X_obs shape: {datas['train_data']['X_obs'].shape}")
    print(f"更新后X_obs_names: {len(datas['train_data']['X_obs_names'])} 个特征")

    # 打印统计信息
    print_youtube_feature_stats(datas)

    # 保存更新后的datas字典（覆盖原文件）
    print(f"\n保存更新后的datas字典到: {data_path}")
    with open(data_path, 'wb') as f:
        pickle.dump(datas, f)

    print("\n测试完成！YouTube特征已添加到datas.pkl")


if __name__ == "__main__":
    main()
