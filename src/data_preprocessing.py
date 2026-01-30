"""
数据预处理模块

处理缺失值的四种类型
"""

import pandas as pd
import numpy as np


def handle_missing_values(config, datas):
    """
    处理缺失值

    四种类型：
    1. 赛季结束后的周（整周N/A）
    2. 只有3个评委（judge4全N/A）
    3. 选手个人缺席且被淘汰（N/A → 移除）
    4. 选手个人缺席未淘汰（N/A → 填充）

    Args:
        config: 配置字典
        datas: 数据字典，包含 'raw_data' 键

    Returns:
        datas: 更新后的数据字典
    """
    df = datas['raw_data'].copy()

    print("=" * 60)
    print("开始处理缺失值")
    print("=" * 60)

    # 步骤1：识别只有3个评委的周
    print("\n[1/3] 识别只有3个评委的周...")
    missing_judge4_weeks = identify_missing_judge4_weeks(df)
    datas['missing_judge4_weeks'] = missing_judge4_weeks

    # 步骤2：识别并处理个人缺席的选手
    print("\n[2/3] 识别并处理个人缺席的选手...")
    df = handle_absent_players(df)

    # 步骤3：验证被淘汰选手的数据
    print("\n[3/3] 验证被淘汰选手的数据...")
    df = verify_eliminated_players(df)

    print("\n" + "=" * 60)
    print("缺失值处理完成")
    print("=" * 60)

    datas['raw_data'] = df
    return datas


def get_week_judge_cols(week):
    """获取某周的评委列名"""
    return [f'week{week}_judge{j}_score' for j in range(1, 5)]


def identify_missing_judge4_weeks(df):
    """
    识别只有3个评委的周

    判定条件：该周所有选手的judge4都是N/A（且该周有比赛）
    """
    missing_judge4_weeks = []
    has_judge4_weeks = []

    # 获取所有周数
    week_cols = [col for col in df.columns if 'week' in col and 'judge1' in col]
    max_week = len(week_cols)

    for season in df['season'].unique():
        season_data = df[df['season'] == season]

        for week in range(1, max_week + 1):
            judge4_col = f'week{week}_judge4_score'

            if judge4_col not in df.columns:
                continue

            # 检查该周是否有比赛（judge1-3有分数）
            judge123_cols = [f'week{week}_judge{j}_score' for j in range(1, 4)]
            has_competition = season_data[judge123_cols].notna().any().any()

            if not has_competition:
                continue

            # 检查该周所有选手的judge4
            judge4_scores = season_data[judge4_col]

            # 如果有比赛，但judge4全是N/A
            if judge4_scores.isna().all():
                missing_judge4_weeks.append((season, week))
            else:
                # 有judge4分数
                has_judge4_weeks.append((season, week))

    print(f"共识别出 {len(missing_judge4_weeks)} 个只有3个评委的周")
    print(f"共识别出 {len(has_judge4_weeks)} 个有4个评委的周")

    # 统计有4个评委的赛季
    seasons_with_judge4 = sorted(set(s for s, w in has_judge4_weeks))
    if seasons_with_judge4:
        print(f"有4个评委的赛季: {seasons_with_judge4}")

    return missing_judge4_weeks


def handle_absent_players(df):
    """
    处理个人缺席的选手

    判定条件：
    1. 某选手某周所有评委评分都是N/A（不是0）
    2. 该周其他选手有评分
    3. 区分已淘汰和未淘汰
    """
    # 获取所有周数
    week_cols = [col for col in df.columns if 'week' in col and 'judge1' in col]
    max_week = len(week_cols)

    filled_count = 0
    removed_count = 0

    for season in df['season'].unique():
        season_mask = df['season'] == season
        season_data = df[season_mask]

        for week in range(1, max_week + 1):
            week_judge_cols = get_week_judge_cols(week)
            week_judge_cols = [col for col in week_judge_cols if col in df.columns]

            if not week_judge_cols:
                continue

            # 检查该周是否有比赛
            has_competition = season_data[week_judge_cols].notna().any().any()

            if not has_competition:
                continue

            # 遍历该周每个选手
            for idx in season_data.index:
                row = df.loc[idx]

                # 检查该选手该周的所有评委评分
                player_scores = [row[col] for col in week_judge_cols]

                # 如果所有评委都是N/A（不是0）
                if all(pd.isna(score) for score in player_scores):
                    celebrity_name = row['celebrity_name']
                    results = row['results']

                    # 检查是否在该周被淘汰
                    is_eliminated = f'Eliminated Week {week}' in str(results)

                    if is_eliminated:
                        # 缺席且被淘汰 → 移除该选手后续所有记录
                        # 将该周及之后的评分设为0（保持数据完整性）
                        for future_week in range(week, max_week + 1):
                            future_cols = get_week_judge_cols(future_week)
                            for col in future_cols:
                                if col in df.columns:
                                    df.at[idx, col] = 0

                        removed_count += 1
                        print(f"  - {celebrity_name} (Season {season}, Week {week}): "
                              f"缺席且被淘汰 → 该周及之后设为0")
                    else:
                        # 缺席但未淘汰 → 填充分数
                        if week == 1:
                            # 第一周缺席 → 填充0
                            fill_value = 0
                            print(f"  - {celebrity_name} (Season {season}, Week {week}): "
                                  f"第一周缺席 → 填充0")
                        else:
                            # 非第一周 → 用本赛季历史均分
                            fill_value = calculate_season_historical_avg(
                                df, idx, season, week
                            )
                            print(f"  - {celebrity_name} (Season {season}, Week {week}): "
                                  f"用本赛季历史均分填充 → {fill_value:.2f}")

                        # 执行填充
                        for col in week_judge_cols:
                            df.at[idx, col] = fill_value

                        filled_count += 1

    print(f"共处理 {filled_count} 个缺席但未淘汰的记录")
    print(f"共处理 {removed_count} 个缺席且被淘汰的记录")
    return df


def calculate_season_historical_avg(df, idx, season, current_week):
    """
    计算选手在本赛季、当前周之前的历史平均分

    Args:
        df: 数据框
        idx: 选手的索引
        season: 赛季
        current_week: 当前周

    Returns:
        历史平均分
    """
    row = df.loc[idx]

    # 计算前几周的平均分
    weekly_scores = []
    for week in range(1, current_week):
        week_judge_cols = get_week_judge_cols(week)
        week_scores = [row[col] for col in week_judge_cols
                      if col in df.columns and pd.notna(row[col]) and row[col] != 0]

        if week_scores:
            weekly_scores.append(np.mean(week_scores))

    if weekly_scores:
        return np.mean(weekly_scores)
    else:
        return 0


def verify_eliminated_players(df):
    """
    验证被淘汰选手的数据

    确保被淘汰后的周分数为0
    如果是N/A，填充为0
    """
    # 获取所有周数
    week_cols = [col for col in df.columns if 'week' in col and 'judge1' in col]
    max_week = len(week_cols)

    fixed_count = 0

    for idx, row in df.iterrows():
        results = row['results']

        # 检查是否被淘汰
        if 'Eliminated Week' in str(results):
            # 提取淘汰周数
            try:
                elimination_week = int(results.split('Week')[1].strip().split()[0])
            except:
                continue

            # 检查淘汰后的周
            for week in range(elimination_week + 1, max_week + 1):
                week_judge_cols = get_week_judge_cols(week)

                for col in week_judge_cols:
                    if col in df.columns:
                        # 如果是N/A，填充为0
                        if pd.isna(row[col]):
                            df.at[idx, col] = 0
                            fixed_count += 1

    if fixed_count > 0:
        print(f"  - 修正了 {fixed_count} 个被淘汰后应为0但是N/A的值")
    else:
        print(f"  - 所有被淘汰选手的数据都正确")

    return df
"""
格式转换模块

将宽格式数据转换为长格式
"""

import pandas as pd

import numpy as np


def convert_to_long_format(config, datas):
    """
    将宽格式数据转换为长格式

    宽格式：每行 = 一个选手在一个赛季的完整表现
    长格式：每行 = 一个选手在某一周的状态

    Args:
        config: 配置字典
        datas: 数据字典

    Returns:
        datas: 更新后的数据字典，添加 'long_data' 键
    """
    df = datas['raw_data'].copy()
    features = datas.get('features', None)

    print("=" * 60)
    print("转换为长格式")
    print("=" * 60)

    # 获取所有周的列
    week_cols = [col for col in df.columns if 'week' in col and 'judge' in col]
    max_week = max([int(col.split('_')[0].replace('week', '')) for col in week_cols])

    print(f"最大周数: {max_week}")

    # 创建长格式数据
    long_rows = []

    for idx, row in df.iterrows():
        celebrity_name = row['celebrity_name']
        season = row['season']
        results = row['results']
        placement = row['placement']

        # 确定该选手的淘汰周（如果有）
        elimination_week = None
        if 'Eliminated Week' in str(results):
            try:
                elimination_week = int(results.split('Week')[1].strip().split()[0])
            except:
                pass

        # 对每一周创建一行
        for week in range(1, max_week + 1):
            # 获取该周的评委分数
            week_judge_cols = [f'week{week}_judge{j}_score' for j in range(1, 5)]
            week_scores = [row[col] for col in week_judge_cols if col in df.columns]

            # 检查该周是否有数据
            has_data = any(pd.notna(score) and score != 0 for score in week_scores)

            # 如果该周没有数据且不是被淘汰后的周，跳过
            if not has_data and (elimination_week is None or week < elimination_week):
                continue

            # 如果是被淘汰后的周，也跳过（不需要这些行）
            if elimination_week is not None and week > elimination_week:
                continue

            # 创建该周的记录
            week_row = {
                'season_id': season,
                'week_id': week,
                'celebrity_name': celebrity_name,
                'placement_final': placement,
            }

            # 添加静态特征
            static_features = [
                'is_male', 'age', 'age_squared', 'age_centered',
                'age_bucket_young', 'age_bucket_mid', 'age_bucket_old',
                'industry_category', 'is_international', 'log_us_state_pop',
                'ballroom_partner'
            ]

            # 添加行业独热编码列
            industry_cols = [col for col in df.columns if col.startswith('industry_')]
            static_features.extend(industry_cols)

            for feat in static_features:
                if feat in row.index:
                    week_row[feat] = row[feat]

            # 添加参赛次数特征（从features中获取）
            if features is not None:
                if 'celebrity_previous_seasons' in features.columns:
                    week_row['celebrity_previous_seasons'] = features.loc[idx, 'celebrity_previous_seasons']
                if 'partner_previous_seasons' in features.columns:
                    week_row['partner_previous_seasons'] = features.loc[idx, 'partner_previous_seasons']
                if 'celebrity_gender' in features.columns:
                    week_row['celebrity_gender'] = features.loc[idx, 'celebrity_gender']
                if 'ballroom_partner_gender' in features.columns:
                    week_row['ballroom_partner_gender'] = features.loc[idx, 'ballroom_partner_gender']

            # 添加该周的评委分数
            for j, col in enumerate(week_judge_cols, 1):
                if col in df.columns:
                    week_row[f'judge{j}_score'] = row[col]

            # 判定该周的result_status
            if elimination_week is not None and week == elimination_week:
                week_row['result_status'] = 1  # Eliminated
            else:
                week_row['result_status'] = 0  # Safe

            long_rows.append(week_row)

    # 创建长格式DataFrame
    long_df = pd.DataFrame(long_rows)

    print(f"\n转换完成:")
    print(f"  - 原始数据: {len(df)} 行（选手-赛季）")
    print(f"  - 长格式数据: {len(long_df)} 行（选手-周）")
    print(f"  - 平均每个选手参赛: {len(long_df) / len(df):.1f} 周")

    # 按season, week, celebrity排序
    long_df = long_df.sort_values(['season_id', 'week_id', 'celebrity_name']).reset_index(drop=True)

    datas['long_data'] = long_df
    return datas
