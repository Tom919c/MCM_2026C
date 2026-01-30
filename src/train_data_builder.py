"""
Train Data Builder Module

Constructs train_data format according to 数据接口规范.md
"""

import pandas as pd
import numpy as np


def build_index_system(long_df):
    """
    建立索引系统

    创建三个映射：
    1. celebrity_name -> celeb_idx (0-based)
    2. ballroom_partner -> pro_idx (0-based)
    3. (season_id, week_id) -> week_idx (0-based, 全局唯一)

    Args:
        long_df: 长格式数据框

    Returns:
        index_maps: 包含所有映射的字典
        long_df: 添加了索引列的数据框
    """
    print("建立索引系统...")

    # 1. Celebrity索引映射
    unique_celebs = sorted(long_df['celebrity_name'].unique())
    celeb_to_idx = {name: idx for idx, name in enumerate(unique_celebs)}
    long_df['celeb_idx'] = long_df['celebrity_name'].map(celeb_to_idx).astype(np.int32)

    print(f"  - 名人总数 (n_celebs): {len(unique_celebs)}")

    # 2. Professional Dancer索引映射
    unique_pros = sorted(long_df['ballroom_partner'].unique())
    pro_to_idx = {name: idx for idx, name in enumerate(unique_pros)}
    long_df['pro_idx'] = long_df['ballroom_partner'].map(pro_to_idx).astype(np.int32)

    print(f"  - 职业舞者总数 (n_pros): {len(unique_pros)}")

    # 3. Week索引映射（全局唯一）
    # 按season_id和week_id排序，分配全局week_idx
    unique_weeks = long_df[['season_id', 'week_id']].drop_duplicates().sort_values(['season_id', 'week_id'])
    week_to_idx = {(row['season_id'], row['week_id']): idx
                   for idx, (_, row) in enumerate(unique_weeks.iterrows())}
    long_df['week_idx'] = long_df.apply(
        lambda row: week_to_idx[(row['season_id'], row['week_id'])], axis=1
    ).astype(np.int32)

    print(f"  - 周总数 (n_weeks): {len(unique_weeks)}")
    print(f"  - 观测总数 (n_obs): {len(long_df)}")

    # 保存映射
    index_maps = {
        'celeb_to_idx': celeb_to_idx,
        'idx_to_celeb': {idx: name for name, idx in celeb_to_idx.items()},
        'pro_to_idx': pro_to_idx,
        'idx_to_pro': {idx: name for name, idx in pro_to_idx.items()},
        'week_to_idx': week_to_idx,
        'idx_to_week': {idx: week for week, idx in week_to_idx.items()},
        'n_celebs': len(unique_celebs),
        'n_pros': len(unique_pros),
        'n_weeks': len(unique_weeks),
        'n_obs': len(long_df)
    }

    return index_maps, long_df


def classify_features():
    """
    特征分类映射

    根据数据接口规范，将特征分为三类：
    - celeb: 名人级别特征（静态，每个名人一个值）
    - pro: 职业舞者级别特征（静态，每个舞者一个值）
    - obs: 观测级别特征（动态，每个选手-周组合一个值）

    Returns:
        feature_classification: 字典，键为特征名，值为分类('celeb', 'pro', 'obs')
    """
    feature_classification = {
        # ========== Celebrity-level features (X_celeb) ==========
        # 性别特征
        'is_male': 'celeb',
        # 'celebrity_gender': 'celeb',  # 字符串，不包含在特征矩阵中

        # 年龄特征
        'age': 'celeb',
        'age_squared': 'celeb',
        'age_centered': 'celeb',
        'age_bucket_young': 'celeb',
        'age_bucket_mid': 'celeb',
        'age_bucket_old': 'celeb',

        # 行业特征（独热编码）
        # 'industry_category': 'celeb',  # 字符串，不包含在特征矩阵中
        'industry_Business, Controversial & Special Roles': 'celeb',
        'industry_Fashion, Beauty & Modeling': 'celeb',
        'industry_Media, News & Communication': 'celeb',
        'industry_Music & Dance': 'celeb',
        'industry_Performing Arts & Entertainment': 'celeb',
        'industry_Public Service & Specialized Professions': 'celeb',
        'industry_Sports & Physical Performance': 'celeb',

        # 地理特征
        'is_international': 'celeb',
        'log_us_state_pop': 'celeb',

        # 历史参赛次数（名人）
        'celebrity_previous_seasons': 'celeb',

        # ========== Professional Dancer-level features (X_pro) ==========
        # 'ballroom_partner_gender': 'pro',  # 字符串，不包含在特征矩阵中
        'partner_previous_seasons': 'pro',
        'pro_prev_wins': 'pro',
        'pro_avg_rank': 'pro',

        # ========== Observation-level features (X_obs) ==========
        # 配对特征
        'same_sex_pair': 'obs',

        # 评委分数（原始）
        'judge1_score': 'obs',
        'judge2_score': 'obs',
        'judge3_score': 'obs',
        'judge4_score': 'obs',
        'judge_score_raw': 'obs',

        # 动态表现特征
        'z_score': 'obs',
        'prev_z_score': 'obs',
        'score_trend': 'obs',
        'is_top_score': 'obs',
        'perfect_score': 'obs',
        'judge_score_stddev': 'obs',

        # 排名特征
        'judge_rank': 'obs',
        'is_bottom_two': 'obs',

        # 历史轨迹特征
        'times_in_bottom': 'obs',
        'cumulative_avg_score': 'obs',
        'teflon_factor': 'obs',
        'weeks_survived': 'obs',

        # 赛制特征
        'season_era': 'obs',
        'voting_rule_type': 'obs',
        'judge_save_active': 'obs',
        'eliminations_this_week': 'obs',

        # 结果标签
        'result_status': 'obs',
    }

    return feature_classification


def extract_feature_matrices(long_df, index_maps, feature_classification):
    """
    从long_df中提取并分离特征矩阵

    根据特征分类，将特征分离为：
    - X_celeb: [n_celebs, ?] 名人特征矩阵
    - X_pro: [n_pros, ?] 职业舞者特征矩阵
    - X_obs: [n_obs, ?] 观测级特征矩阵

    Args:
        long_df: 长格式数据框（包含所有特征）
        index_maps: 索引映射字典
        feature_classification: 特征分类字典

    Returns:
        feature_matrices: 包含X_celeb, X_pro, X_obs及其名称列表的字典
    """
    print("提取并分离特征矩阵...")

    n_celebs = index_maps['n_celebs']
    n_pros = index_maps['n_pros']
    n_obs = index_maps['n_obs']

    # 分类特征名称
    celeb_features = [f for f, cat in feature_classification.items() if cat == 'celeb' and f in long_df.columns]
    pro_features = [f for f, cat in feature_classification.items() if cat == 'pro' and f in long_df.columns]
    obs_features = [f for f, cat in feature_classification.items() if cat == 'obs' and f in long_df.columns]

    print(f"  - Celebrity特征数: {len(celeb_features)}")
    print(f"  - Professional Dancer特征数: {len(pro_features)}")
    print(f"  - Observation特征数: {len(obs_features)}")

    # ========== 提取X_celeb ==========
    # 对每个名人，取第一次出现的特征值（因为是静态特征）
    celeb_data = long_df.groupby('celeb_idx')[celeb_features].first()
    celeb_data = celeb_data.sort_index()  # 确保按celeb_idx排序
    X_celeb = celeb_data.values.astype(np.float32)

    # ========== 提取X_pro ==========
    # 对每个职业舞者，取第一次出现的特征值（因为是静态特征）
    pro_data = long_df.groupby('pro_idx')[pro_features].first()
    pro_data = pro_data.sort_index()  # 确保按pro_idx排序
    X_pro = pro_data.values.astype(np.float32)

    # ========== 提取X_obs ==========
    # 观测级特征直接提取（每行一个观测）
    X_obs = long_df[obs_features].values.astype(np.float32)

    # 验证形状
    assert X_celeb.shape == (n_celebs, len(celeb_features)), \
        f"X_celeb shape mismatch: {X_celeb.shape} vs ({n_celebs}, {len(celeb_features)})"
    assert X_pro.shape == (n_pros, len(pro_features)), \
        f"X_pro shape mismatch: {X_pro.shape} vs ({n_pros}, {len(pro_features)})"
    assert X_obs.shape == (n_obs, len(obs_features)), \
        f"X_obs shape mismatch: {X_obs.shape} vs ({n_obs}, {len(obs_features)})"

    print(f"  - X_celeb shape: {X_celeb.shape}")
    print(f"  - X_pro shape: {X_pro.shape}")
    print(f"  - X_obs shape: {X_obs.shape}")

    feature_matrices = {
        'X_celeb': X_celeb,
        'X_pro': X_pro,
        'X_obs': X_obs,
        'X_celeb_names': celeb_features,
        'X_pro_names': pro_features,
        'X_obs_names': obs_features
    }

    return feature_matrices


def build_week_data(long_df, index_maps):
    """
    构建周级结构数据

    为每一周创建一个字典，包含：
    - obs_mask: 该周参赛选手的掩码 [n_obs]
    - n_contestants: 该周参赛人数
    - n_eliminated: 该周淘汰人数
    - eliminated_mask: 被淘汰选手的掩码 [n_obs]
    - rule_method: 投票规则 (0=排名法, 1=百分比法)
    - judge_save_active: 是否有评委拯救环节

    Args:
        long_df: 长格式数据框
        index_maps: 索引映射字典

    Returns:
        week_data: list[dict], 长度为n_weeks
    """
    print("构建周级结构数据...")

    n_weeks = index_maps['n_weeks']
    n_obs = index_maps['n_obs']

    week_data = []

    for week_idx in range(n_weeks):
        # 获取该周的所有观测
        week_mask = (long_df['week_idx'] == week_idx).values

        # 该周参赛人数
        n_contestants = week_mask.sum()

        # 该周淘汰人数
        eliminated_mask = (long_df['week_idx'] == week_idx) & (long_df['result_status'] == 1)
        n_eliminated = eliminated_mask.sum()

        # 获取该周的投票规则和评委拯救信息（取第一个观测的值）
        week_rows = long_df[week_mask]
        if len(week_rows) > 0:
            rule_method = int(week_rows['voting_rule_type'].iloc[0])
            judge_save_active = bool(week_rows['judge_save_active'].iloc[0])
        else:
            rule_method = 0
            judge_save_active = False

        week_dict = {
            'obs_mask': week_mask,
            'n_contestants': int(n_contestants),
            'n_eliminated': int(n_eliminated),
            'eliminated_mask': eliminated_mask.values,
            'rule_method': rule_method,
            'judge_save_active': judge_save_active
        }

        week_data.append(week_dict)

    print(f"  - 周级数据构建完成: {len(week_data)} 周")

    return week_data


def calculate_elimination_rule_data(long_df, missing_judge4_weeks):
    """
    计算淘汰规则相关数据

    根据数据接口规范，计算：
    1. judge_score_pct: 评委分占当周总分的比例（百分比法用）
    2. judge_rank_score: 评委排名分（排名法用）

    Args:
        long_df: 长格式数据框
        missing_judge4_weeks: 只有3个评委的周列表 [(season, week), ...]

    Returns:
        long_df: 添加了judge_score_pct和judge_rank_score列的数据框
    """
    print("计算淘汰规则相关数据...")

    # ========== 1. 计算judge_score_pct（百分比法） ==========
    # judge_score_pct = 评委分 / 当周最高评委分
    def calc_score_pct(group):
        group = group.copy()
        max_score = group['judge_score_raw'].max()
        if max_score > 0:
            group['judge_score_pct'] = group['judge_score_raw'] / max_score
        else:
            group['judge_score_pct'] = 0.0
        return group

    long_df = long_df.groupby(['season_id', 'week_id'], group_keys=False).apply(calc_score_pct)

    # ========== 2. 计算judge_rank_score（排名法） ==========
    # judge_rank_score = (n_contestants - rank) / (n_contestants - 1)
    # 其中rank是评委分排名（1=最高分）
    def calc_rank_score(group):
        group = group.copy()
        n_contestants = len(group)
        # judge_rank已经在特征工程中计算过了
        if n_contestants > 1:
            group['judge_rank_score'] = (n_contestants - group['judge_rank']) / (n_contestants - 1)
        else:
            group['judge_rank_score'] = 1.0
        return group

    long_df = long_df.groupby(['season_id', 'week_id'], group_keys=False).apply(calc_rank_score)

    print(f"  - judge_score_pct: 范围=[{long_df['judge_score_pct'].min():.3f}, {long_df['judge_score_pct'].max():.3f}]")
    print(f"  - judge_rank_score: 范围=[{long_df['judge_rank_score'].min():.3f}, {long_df['judge_rank_score'].max():.3f}]")

    return long_df


def standardize_features(train_data):
    """
    标准化连续特征

    根据数据接口规范，所有连续特征需标准化（均值0，标准差1）

    Args:
        train_data: train_data字典

    Returns:
        train_data: 标准化后的train_data
        standardization_params: 标准化参数（用于逆变换）
    """
    print("标准化连续特征...")

    standardization_params = {}

    # 需要标准化的特征（排除二值特征和已标准化的特征）
    # Celebrity features
    celeb_continuous = ['age', 'age_squared', 'age_centered', 'log_us_state_pop', 'celebrity_previous_seasons']
    # Pro features
    pro_continuous = ['partner_previous_seasons', 'pro_prev_wins', 'pro_avg_rank']
    # Obs features
    obs_continuous = [
        'judge1_score', 'judge2_score', 'judge3_score', 'judge4_score',
        'judge_score_raw', 'z_score', 'prev_z_score', 'score_trend',
        'judge_score_stddev', 'judge_rank', 'times_in_bottom',
        'cumulative_avg_score', 'weeks_survived'
    ]

    def standardize_matrix(X, feature_names, continuous_features):
        """标准化矩阵中的连续特征"""
        X_std = X.copy()
        params = {}

        for feat in continuous_features:
            if feat in feature_names:
                idx = feature_names.index(feat)
                col = X[:, idx]

                # 计算均值和标准差
                mean = col.mean()
                std = col.std()

                # 标准化（避免除以0）
                if std > 1e-8:
                    X_std[:, idx] = (col - mean) / std
                else:
                    X_std[:, idx] = col - mean

                params[feat] = {'mean': float(mean), 'std': float(std)}

        return X_std, params

    # 标准化X_celeb
    X_celeb_std, celeb_params = standardize_matrix(
        train_data['X_celeb'],
        train_data['X_celeb_names'],
        celeb_continuous
    )
    train_data['X_celeb'] = X_celeb_std
    standardization_params['celeb'] = celeb_params

    # 标准化X_pro
    X_pro_std, pro_params = standardize_matrix(
        train_data['X_pro'],
        train_data['X_pro_names'],
        pro_continuous
    )
    train_data['X_pro'] = X_pro_std
    standardization_params['pro'] = pro_params

    # 标准化X_obs
    X_obs_std, obs_params = standardize_matrix(
        train_data['X_obs'],
        train_data['X_obs_names'],
        obs_continuous
    )
    train_data['X_obs'] = X_obs_std
    standardization_params['obs'] = obs_params

    print(f"  - 标准化完成: {len(celeb_params)} celeb特征, {len(pro_params)} pro特征, {len(obs_params)} obs特征")

    return train_data, standardization_params


def build_train_data(config, datas):
    """
    构建train_data格式

    将long_data转换为符合数据接口规范的train_data格式

    Args:
        config: 配置字典
        datas: 数据字典（包含long_data）

    Returns:
        datas: 更新后的数据字典（添加train_data）
    """
    print("\n" + "=" * 60)
    print("构建train_data格式")
    print("=" * 60)

    long_df = datas['long_data'].copy()
    missing_judge4_weeks = datas.get('missing_judge4_weeks', [])

    # Stage 1: 建立索引系统
    print("\n[Stage 1/6] 建立索引系统...")
    index_maps, long_df = build_index_system(long_df)

    # Stage 2 & 3: 特征分类和提取
    print("\n[Stage 2/6] 特征分类...")
    feature_classification = classify_features()

    print("\n[Stage 3/6] 提取特征矩阵...")
    feature_matrices = extract_feature_matrices(long_df, index_maps, feature_classification)

    # Stage 4: 构建周级结构
    print("\n[Stage 4/6] 构建周级结构...")
    week_data = build_week_data(long_df, index_maps)

    # Stage 5: 计算淘汰规则数据
    print("\n[Stage 5/6] 计算淘汰规则数据...")
    long_df = calculate_elimination_rule_data(long_df, missing_judge4_weeks)

    # Stage 6: 组装train_data
    print("\n[Stage 6/6] 组装train_data...")
    train_data = {
        # 维度信息
        'n_obs': index_maps['n_obs'],
        'n_weeks': index_maps['n_weeks'],
        'n_celebs': index_maps['n_celebs'],
        'n_pros': index_maps['n_pros'],

        # 索引数组
        'celeb_idx': long_df['celeb_idx'].values.astype(np.int32),
        'pro_idx': long_df['pro_idx'].values.astype(np.int32),
        'week_idx': long_df['week_idx'].values.astype(np.int32),

        # 特征矩阵
        'X_celeb': feature_matrices['X_celeb'],
        'X_pro': feature_matrices['X_pro'],
        'X_obs': feature_matrices['X_obs'],

        # 特征名称
        'X_celeb_names': feature_matrices['X_celeb_names'],
        'X_pro_names': feature_matrices['X_pro_names'],
        'X_obs_names': feature_matrices['X_obs_names'],

        # 淘汰规则相关
        'judge_score_pct': long_df['judge_score_pct'].values.astype(np.float32),
        'judge_rank_score': long_df['judge_rank_score'].values.astype(np.float32),

        # 周级数据
        'week_data': week_data
    }

    print(f"\ntrain_data构建完成:")
    print(f"  - 维度: n_obs={train_data['n_obs']}, n_weeks={train_data['n_weeks']}, "
          f"n_celebs={train_data['n_celebs']}, n_pros={train_data['n_pros']}")
    print(f"  - 特征矩阵: X_celeb{train_data['X_celeb'].shape}, "
          f"X_pro{train_data['X_pro'].shape}, X_obs{train_data['X_obs'].shape}")

    # Stage 6: 标准化特征
    print("\n[Stage 6.5/6] 标准化特征...")
    train_data, standardization_params = standardize_features(train_data)

    # 更新datas
    datas['train_data'] = train_data
    datas['long_data'] = long_df  # 更新long_df（添加了新列）
    datas['index_maps'] = index_maps
    datas['standardization_params'] = standardization_params

    return datas
