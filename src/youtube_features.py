"""
YouTube Features Module

从YouTube爬虫数据中提取特征，添加到训练数据中
"""

import pandas as pd
import numpy as np
import re
import os


def extract_week_from_title(title):
    """
    从视频标题中提取周次信息

    支持多种格式：
    - "Week 3"
    - "Week-3"
    - "week 3"
    - "DWTS Week 3"
    - "Week3"
    - "Wk 3"
    - "W3"
    - "3rd week"
    - "Semifinals" (半决赛，通常是倒数第二周)
    - "Finals" (决赛，最后一周)

    Args:
        title: 视频标题字符串

    Returns:
        week_num: 周次（整数），如果无法提取则返回None
    """
    if pd.isna(title) or not isinstance(title, str):
        return None

    title_lower = title.lower()

    # 模式1: "Week 3", "Week-3", "Week3", "week 3"
    match = re.search(r'week[\s-]?(\d+)', title_lower)
    if match:
        return int(match.group(1))

    # 模式2: "Wk 3", "Wk3"
    match = re.search(r'wk[\s-]?(\d+)', title_lower)
    if match:
        return int(match.group(1))

    # 模式3: "W3" (单独的W后面跟数字)
    match = re.search(r'\bw(\d+)\b', title_lower)
    if match:
        return int(match.group(1))

    # 模式4: "3rd week", "2nd week", "1st week"
    match = re.search(r'(\d+)(?:st|nd|rd|th)\s+week', title_lower)
    if match:
        return int(match.group(1))

    # 模式5: 特殊周次
    if 'semifinal' in title_lower or 'semi-final' in title_lower or 'semi final' in title_lower:
        return -1  # 标记为半决赛，后续需要根据赛季信息确定具体周次

    if 'final' in title_lower and 'semifinal' not in title_lower:
        return -2  # 标记为决赛，后续需要根据赛季信息确定具体周次

    return None


def load_youtube_data(youtube_data_path):
    """
    读取YouTube爬虫数据并提取周次信息

    Args:
        youtube_data_path: YouTube数据文件路径

    Returns:
        youtube_df: 包含周次信息的DataFrame
    """
    print("加载YouTube数据...")

    # 使用多编码尝试机制读取CSV
    encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252', 'gbk']
    youtube_df = None
    for encoding in encodings:
        try:
            youtube_df = pd.read_csv(youtube_data_path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue

    if youtube_df is None:
        youtube_df = pd.read_csv(youtube_data_path, encoding='utf-8', errors='ignore')

    print(f"  - 加载了 {len(youtube_df)} 条YouTube视频记录")

    # 提取周次信息
    youtube_df['week_num'] = youtube_df['yt_video_title'].apply(extract_week_from_title)

    # 筛选出能确定周次的视频
    youtube_with_week = youtube_df[youtube_df['week_num'].notna()].copy()

    print(f"  - 其中 {len(youtube_with_week)} 条记录能确定周次")
    print(f"  - 覆盖率: {len(youtube_with_week) / len(youtube_df) * 100:.2f}%")

    # 处理特殊周次（半决赛、决赛）
    # 这里暂时保留-1和-2标记，后续可以根据赛季信息进一步处理

    return youtube_with_week


def aggregate_youtube_by_celeb_season_week(youtube_df):
    """
    按名人-赛季-周次聚合YouTube数据

    如果同一个名人在同一赛季同一周有多个视频，取平均值

    Args:
        youtube_df: 包含周次信息的YouTube DataFrame

    Returns:
        aggregated_df: 聚合后的DataFrame
    """
    print("按名人-赛季-周次聚合YouTube数据...")

    # 过滤掉特殊周次标记（-1, -2）
    youtube_df = youtube_df[youtube_df['week_num'] > 0].copy()

    # 按名人-赛季-周次分组，计算平均值
    aggregated = youtube_df.groupby(['celebrity_name', 'season', 'week_num']).agg({
        'yt_view_count': 'mean',
        'yt_like_count': 'mean',
        'yt_comment_count': 'mean'
    }).reset_index()

    print(f"  - 聚合后得到 {len(aggregated)} 个唯一的名人-赛季-周次组合")

    return aggregated


def normalize_youtube_features(aggregated_df):
    """
    使用Z-score标准化YouTube特征

    Args:
        aggregated_df: 聚合后的DataFrame

    Returns:
        aggregated_df: 添加了归一化特征的DataFrame
        normalization_params: 归一化参数（均值和标准差）
    """
    print("使用Z-score标准化YouTube特征...")

    normalization_params = {}

    for col in ['yt_view_count', 'yt_like_count', 'yt_comment_count']:
        mean = aggregated_df[col].mean()
        std = aggregated_df[col].std()

        # 避免除以0
        if std > 1e-8:
            aggregated_df[f'{col}_norm'] = (aggregated_df[col] - mean) / std
        else:
            aggregated_df[f'{col}_norm'] = aggregated_df[col] - mean

        normalization_params[col] = {'mean': float(mean), 'std': float(std)}

        print(f"  - {col}: mean={mean:.2f}, std={std:.2f}")

    return aggregated_df, normalization_params


def add_youtube_features_to_train_data(config, datas):
    """
    将YouTube特征添加到train_data中

    在X_obs中添加4个新特征：
    1. has_youtube_video: 二值特征，是否有YouTube视频数据
    2. youtube_view_count_norm: 归一化的播放量
    3. youtube_like_count_norm: 归一化的点赞数
    4. youtube_comment_count_norm: 归一化的评论数

    Args:
        config: 配置字典
        datas: 数据字典（包含train_data和long_data）

    Returns:
        datas: 更新后的数据字典
    """
    print("\n" + "=" * 60)
    print("添加YouTube特征到train_data")
    print("=" * 60)

    # 获取YouTube数据路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    youtube_data_path = os.path.join(project_root, 'data', 'processed', 'youtube_data.csv')

    if not os.path.exists(youtube_data_path):
        print(f"警告: YouTube数据文件不存在: {youtube_data_path}")
        print("跳过YouTube特征添加")
        return datas

    # Step 1: 加载YouTube数据并提取周次
    youtube_with_week = load_youtube_data(youtube_data_path)

    if len(youtube_with_week) == 0:
        print("警告: 没有找到包含周次信息的YouTube视频")
        print("跳过YouTube特征添加")
        return datas

    # Step 2: 按名人-赛季-周次聚合
    aggregated_youtube = aggregate_youtube_by_celeb_season_week(youtube_with_week)

    # Step 3: Z-score标准化
    aggregated_youtube, normalization_params = normalize_youtube_features(aggregated_youtube)

    # Step 4: 匹配到train_data
    print("\n匹配YouTube特征到train_data...")

    train_data = datas['train_data']
    long_data = datas['long_data']

    n_obs = train_data['n_obs']

    # 初始化4个新特征（全部为0）
    has_youtube_video = np.zeros(n_obs, dtype=np.float32)
    youtube_view_count_norm = np.zeros(n_obs, dtype=np.float32)
    youtube_like_count_norm = np.zeros(n_obs, dtype=np.float32)
    youtube_comment_count_norm = np.zeros(n_obs, dtype=np.float32)

    # 创建匹配字典：(celebrity_name, season, week_id) -> (view_norm, like_norm, comment_norm)
    youtube_dict = {}
    for _, row in aggregated_youtube.iterrows():
        key = (row['celebrity_name'], int(row['season']), int(row['week_num']))
        youtube_dict[key] = (
            row['yt_view_count_norm'],
            row['yt_like_count_norm'],
            row['yt_comment_count_norm']
        )

    # 遍历long_data，匹配YouTube特征
    matched_count = 0
    for idx, row in long_data.iterrows():
        celebrity_name = row['celebrity_name']
        season = int(row['season_id'])
        week_id = int(row['week_id'])

        key = (celebrity_name, season, week_id)

        if key in youtube_dict:
            view_norm, like_norm, comment_norm = youtube_dict[key]

            has_youtube_video[idx] = 1.0
            youtube_view_count_norm[idx] = view_norm
            youtube_like_count_norm[idx] = like_norm
            youtube_comment_count_norm[idx] = comment_norm

            matched_count += 1

    print(f"  - 成功匹配 {matched_count} 个观测（占比: {matched_count / n_obs * 100:.2f}%）")

    # Step 5: 添加到X_obs
    print("\n添加特征到X_obs...")

    # 将4个新特征添加到X_obs矩阵
    new_features = np.column_stack([
        has_youtube_video,
        youtube_view_count_norm,
        youtube_like_count_norm,
        youtube_comment_count_norm
    ])

    train_data['X_obs'] = np.hstack([train_data['X_obs'], new_features])

    # 更新X_obs_names
    train_data['X_obs_names'].extend([
        'has_youtube_video',
        'youtube_view_count_norm',
        'youtube_like_count_norm',
        'youtube_comment_count_norm'
    ])

    print(f"  - X_obs shape: {train_data['X_obs'].shape}")
    print(f"  - X_obs_names: {len(train_data['X_obs_names'])} 个特征")

    # 保存归一化参数
    datas['youtube_normalization_params'] = normalization_params

    # 保存聚合后的YouTube数据（用于分析）
    datas['aggregated_youtube'] = aggregated_youtube

    print("\nYouTube特征添加完成！")
    print("=" * 60)

    return datas
