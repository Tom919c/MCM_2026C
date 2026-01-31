"""
特征工程模块

提供特征构建功能
"""

import pandas as pd
import numpy as np
import gender_guesser.detector as gender


def add_gender_features(config, datas):
    """
    为名人和职业舞者添加性别识别特征

    使用gender-guesser库根据名字推测性别，然后应用手动修正

    Args:
        config: 配置字典
        datas: 数据字典，包含 'raw_data' 键

    Returns:
        datas: 更新后的数据字典，在 'features' 中添加性别特征列
    """
    df = datas['raw_data']

    print("正在识别名人和职业舞者的性别...")

    # 初始化性别检测器
    detector = gender.Detector()

    def extract_first_name(full_name):
        """提取名字的第一部分"""
        if pd.isna(full_name):
            return None
        # 移除常见的称谓
        name = str(full_name).strip()
        # 提取第一个名字
        first_name = name.split()[0] if name else None
        return first_name

    def guess_gender(full_name):
        """根据名字推测性别"""
        first_name = extract_first_name(full_name)
        if first_name is None:
            return 'unknown'
        return detector.get_gender(first_name)

    # 使用向量化方法识别性别
    celebrity_gender = df['celebrity_name'].apply(guess_gender)
    ballroom_partner_gender = df['ballroom_partner'].apply(guess_gender)

    # 归类性别值：mostly_female/mostly_male → female/male, andy/unknown → unknown
    gender_mapping = {
        'mostly_female': 'female',
        'mostly_male': 'male',
        'andy': 'unknown'
    }
    celebrity_gender = celebrity_gender.replace(gender_mapping)
    ballroom_partner_gender = ballroom_partner_gender.replace(gender_mapping)

    print("自动性别识别完成")

    # 手动修正性别信息（来源：Wikipedia & official interviews）
    print("正在应用手动性别修正...")

    # 名人性别信息
    celebrity_gender_corrections = {
        "AJ McLean": "male", "Alek Skarlatos": "male", "Apolo Anton Ohno": "male",
        "Arike Ogunbowale": "female", "Audrina Patridge": "female", "Babyface": "male",
        "Betsey Johnson": "female", "Bindi Irwin": "female", "Bonner Bolton": "male",
        "Bristol Palin": "female", "Chaka Khan": "female", "Charli D'Amelio": "female",
        "Charo": "female", "Chrishell Stause": "female", "Chynna Phillips": "female",
        "Cloris Leachman": "female", "D. L. Hughley": "male", "Frankie Muniz": "male",
        "Gabby Windey": "female", "Hayes Grier": "male", "Hines Ward": "male",
        "J.R. Martinez": "male", "JoJo Siwa": "female", "Jordin Sparks": "female",
        "Karamo Brown": "male", "Kel Mitchell": "male", "Leeza Gibbons": "female",
        "Lil' Kim": "female", "Master P": "male", "Melora Hardin": "female",
        "Mirai Nagasu": "female", "Mr. T": "male", "Nastia Liukin": "female",
        "NeNe Leakes": "female", "Nev Schulman": "male", "Niecy Nash": "female",
        "Normani": "female", "Nyle DiMarco": "male", "Paige VanZant": "female",
        "Penn Jillette": "male", "Redfoo": "male", "Riker Lynch": "male",
        "Roshon Fegan": "male", "Rumer Willis": "female", "Sailor Brinkley-Cook": "female",
        "Shandi Finnessey": "female", "Shangela": "male", "Skai Jackson": "female",
        "Steve-O": "male", "Sugar Ray Leonard": "male", "Suni Lee": "female",
        "Tinashe": "female", "Vanilla Ice": "male", "Vinny Guadagnino": "male",
        "Vivica A. Fox": "female", "Von Miller": "male", "Wanya Morris": "male",
        "Wynonna Judd": "female", "Zendaya": "female"
    }

    # 职业舞者性别信息
    partner_gender_corrections = {
        "Corky Ballas": "male", "Keo Motsepe": "male", "Sharna Burgess": "female",
        "Tyne Stecklein": "female", "Witney Carson": "female"
    }

    # 应用名人性别修正
    celebrity_corrections_count = 0
    for name, correct_gender in celebrity_gender_corrections.items():
        mask = (df['celebrity_name'] == name) & (celebrity_gender == 'unknown')
        if mask.any():
            celebrity_gender.loc[mask] = correct_gender
            celebrity_corrections_count += mask.sum()

    # 应用职业舞者性别修正
    partner_corrections_count = 0
    for name, correct_gender in partner_gender_corrections.items():
        # 处理特殊情况：Witney Carson (Xoshitl Gomez week 9)
        if name == "Witney Carson":
            mask = (df['ballroom_partner'].str.contains('Witney Carson', na=False)) & \
                   (ballroom_partner_gender == 'unknown')
        else:
            mask = (df['ballroom_partner'] == name) & (ballroom_partner_gender == 'unknown')

        if mask.any():
            ballroom_partner_gender.loc[mask] = correct_gender
            partner_corrections_count += mask.sum()

    if celebrity_corrections_count > 0 or partner_corrections_count > 0:
        print(f"  - 修正名人性别: {celebrity_corrections_count} 条记录")
        print(f"  - 修正职业舞者性别: {partner_corrections_count} 条记录")

    print("性别识别完成")
    print(f"名人性别分布:")
    print(celebrity_gender.value_counts())
    print(f"\n职业舞者性别分布:")
    print(ballroom_partner_gender.value_counts())

    # 添加到特征DataFrame
    if 'features' not in datas:
        datas['features'] = pd.DataFrame(index=df.index)

    datas['features']['celebrity_gender'] = celebrity_gender.values
    datas['features']['ballroom_partner_gender'] = ballroom_partner_gender.values

    return datas


def add_previous_seasons_count(config, datas):
    """
    为名人和职业舞者添加历史参赛次数特征

    计算名人和职业舞者在参加当前赛季之前参加过几季的比赛

    Args:
        config: 配置字典
        datas: 数据字典，包含 'raw_data' 键

    Returns:
        datas: 更新后的数据字典，在 'features' 中添加特征列
    """
    df = datas['raw_data'].copy()

    print("正在计算历史参赛次数...")

    # ========== 1. 计算名人历史参赛次数 ==========
    print("\n[1/2] 名人历史参赛次数:")

    # 按名人姓名和赛季排序
    df_celebrity = df.sort_values(['celebrity_name', 'season']).reset_index(drop=True)

    # 计算每个名人在当前赛季之前出现的次数
    celebrity_previous_seasons = df_celebrity.groupby('celebrity_name').cumcount()

    print(f"  - 首次参赛: {(celebrity_previous_seasons == 0).sum()} 人次")
    print(f"  - 二次参赛: {(celebrity_previous_seasons == 1).sum()} 人次")
    print(f"  - 三次及以上: {(celebrity_previous_seasons >= 2).sum()} 人次")
    print(f"  - 最多参赛次数: {celebrity_previous_seasons.max()} 次")

    # 统计参加过多次的名人
    celebrity_multi = df_celebrity[celebrity_previous_seasons > 0]['celebrity_name'].unique()
    if len(celebrity_multi) > 0:
        print(f"  - 参加过多次的名人: {len(celebrity_multi)} 人")

    # ========== 2. 计算职业舞者历史参赛次数 ==========
    print("\n[2/2] 职业舞者历史参赛次数:")

    # 按职业舞者姓名和赛季排序
    df_partner = df.sort_values(['ballroom_partner', 'season']).reset_index(drop=True)

    # 计算每个职业舞者在当前赛季之前出现的次数
    partner_previous_seasons = df_partner.groupby('ballroom_partner').cumcount()

    print(f"  - 首次参赛: {(partner_previous_seasons == 0).sum()} 人次")
    print(f"  - 二次参赛: {(partner_previous_seasons == 1).sum()} 人次")
    print(f"  - 三次及以上: {(partner_previous_seasons >= 2).sum()} 人次")
    print(f"  - 最多参赛次数: {partner_previous_seasons.max()} 次")

    # 统计参加过多次的职业舞者
    partner_multi = df_partner[partner_previous_seasons > 0]['ballroom_partner'].unique()
    if len(partner_multi) > 0:
        print(f"  - 参加过多次的职业舞者: {len(partner_multi)} 人")

    # ========== 3. 保存特征 ==========
    # 初始化或更新特征DataFrame
    if 'features' not in datas:
        datas['features'] = pd.DataFrame(index=df.index)

    # 将排序后的数据恢复到原始索引顺序
    celebrity_previous_seasons_aligned = pd.Series(
        celebrity_previous_seasons.values,
        index=df_celebrity.index
    ).reindex(df.index)

    partner_previous_seasons_aligned = pd.Series(
        partner_previous_seasons.values,
        index=df_partner.index
    ).reindex(df.index)

    datas['features']['celebrity_previous_seasons'] = celebrity_previous_seasons_aligned.values
    datas['features']['partner_previous_seasons'] = partner_previous_seasons_aligned.values

    # 保存统计信息到datas
    datas['celebrity_participation_stats'] = {
        'first_time': (celebrity_previous_seasons == 0).sum(),
        'second_time': (celebrity_previous_seasons == 1).sum(),
        'three_plus': (celebrity_previous_seasons >= 2).sum(),
        'max_count': int(celebrity_previous_seasons.max()),
        'multi_participants': celebrity_multi.tolist() if len(celebrity_multi) > 0 else []
    }

    datas['partner_participation_stats'] = {
        'first_time': (partner_previous_seasons == 0).sum(),
        'second_time': (partner_previous_seasons == 1).sum(),
        'three_plus': (partner_previous_seasons >= 2).sum(),
        'max_count': int(partner_previous_seasons.max()),
        'multi_participants': partner_multi.tolist() if len(partner_multi) > 0 else []
    }

    print("\n历史参赛次数计算完成")

    return datas
"""
静态特征模块

构建选手的静态特征（不随周变化）
"""

import pandas as pd
import numpy as np


# 行业归类映射表（根据特征工程文档）
INDUSTRY_MAPPING = {
    'Actor/Actress': 'Performing Arts & Entertainment',
    'Athlete': 'Sports & Physical Performance',
    'Model': 'Fashion, Beauty & Modeling',
    'Singer/Rapper': 'Music & Dance',
    'TV Personality': 'Performing Arts & Entertainment',
    'News Anchor': 'Media, News & Communication',
    'Sports Broadcaster': 'Sports & Physical Performance',
    'Beauty Pagent': 'Fashion, Beauty & Modeling',
    'Racing Driver': 'Sports & Physical Performance',
    'Magician': 'Performing Arts & Entertainment',
    'Radio Personality': 'Performing Arts & Entertainment',
    'Comedian': 'Performing Arts & Entertainment',
    'Entrepreneur': 'Business, Controversial & Special Roles',
    'Politician': 'Public Service & Specialized Professions',
    'Astronaut': 'Public Service & Specialized Professions',
    'Fashion Designer': 'Performing Arts & Entertainment',
    'Social Media Personality': 'Media, News & Communication',
    'Motivational Speaker': 'Music & Dance',
    'Military': 'Public Service & Specialized Professions',
    'Journalist': 'Media, News & Communication',
    'Musician': 'Music & Dance',
    'Producer': 'Performing Arts & Entertainment',
    'Fitness Instructor': 'Sports & Physical Performance',
    'Con artist': 'Business, Controversial & Special Roles',
    'Conservationist': 'Public Service & Specialized Professions',
    'Social media personality': 'Media, News & Communication'
}


def add_static_features(config, datas):
    """
    添加选手静态特征

    包括：
    - 性别特征（已有）
    - 年龄特征
    - 行业特征
    - 地理特征
    - 州人口特征

    Args:
        config: 配置字典
        datas: 数据字典

    Returns:
        datas: 更新后的数据字典
    """
    df = datas['raw_data'].copy()

    print("=" * 60)
    print("添加静态特征")
    print("=" * 60)

    # 1. 性别特征（已有celebrity_gender）
    print("\n[1/5] 处理性别特征...")
    df = add_gender_features_static(df, datas)

    # 2. 年龄特征
    print("\n[2/5] 添加年龄特征...")
    df = add_age_features(df)

    # 3. 行业特征
    print("\n[3/5] 添加行业特征...")
    df = add_industry_features(df)

    # 4. 地理特征
    print("\n[4/5] 添加地理特征...")
    df = add_geography_features(df)

    # 5. 州人口特征
    print("\n[5/5] 添加州人口特征...")
    df = add_state_population_features(df, config)

    print("\n" + "=" * 60)
    print("静态特征添加完成")
    print("=" * 60)

    datas['raw_data'] = df
    return datas


def add_gender_features_static(df, datas):
    """
    添加性别特征（从features中获取）

    特征：
    - is_male: 名人为男性=1，女性=0
    """
    # 从features中获取性别信息
    if 'features' in datas:
        features = datas['features']
        df['is_male'] = (features['celebrity_gender'] == 'male').astype(int)
        print(f"  - is_male: {df['is_male'].sum()} 男性, {(~df['is_male'].astype(bool)).sum()} 女性")
    else:
        print("  - 警告：features数据不存在，跳过性别特征")

    return df


def add_age_features(df):
    """
    添加年龄特征

    特征：
    - age: 原始年龄
    - age_squared: 年龄平方
    - age_centered: 中心化年龄
    - age_bucket_young: 年龄<25
    - age_bucket_mid: 25<=年龄<=50
    - age_bucket_old: 年龄>50
    """
    # 原始年龄
    df['age'] = df['celebrity_age_during_season']

    # 年龄平方
    df['age_squared'] = df['age'] ** 2

    # 年龄中心化
    age_mean = df['age'].mean()
    df['age_centered'] = df['age'] - age_mean

    # 年龄分桶
    df['age_bucket_young'] = (df['age'] < 25).astype(int)
    df['age_bucket_mid'] = ((df['age'] >= 25) & (df['age'] <= 50)).astype(int)
    df['age_bucket_old'] = (df['age'] > 50).astype(int)

    print(f"  - age: 均值={age_mean:.2f}, 范围=[{df['age'].min()}, {df['age'].max()}]")
    print(f"  - age_bucket_young: {df['age_bucket_young'].sum()} 人")
    print(f"  - age_bucket_mid: {df['age_bucket_mid'].sum()} 人")
    print(f"  - age_bucket_old: {df['age_bucket_old'].sum()} 人")

    return df


def add_industry_features(df):
    """
    添加行业特征

    将celebrity_industry归类为7大类并独热编码
    """
    # 归类
    df['industry_category'] = df['celebrity_industry'].map(INDUSTRY_MAPPING)

    # 处理未映射的行业
    unmapped = df[df['industry_category'].isna()]['celebrity_industry'].unique()
    if len(unmapped) > 0:
        print(f"  - 警告：以下行业未映射，归为Other: {unmapped}")
        df['industry_category'] = df['industry_category'].fillna('Other')

    # 独热编码
    industry_dummies = pd.get_dummies(df['industry_category'], prefix='industry')
    df = pd.concat([df, industry_dummies], axis=1)

    print(f"  - 行业分布:")
    for col in industry_dummies.columns:
        print(f"    {col}: {industry_dummies[col].sum()} 人")

    return df


def add_geography_features(df):
    """
    添加地理特征

    特征：
    - is_international: 非美国选手=1
    """
    df['is_international'] = (
        df['celebrity_homecountry/region'] != 'United States'
    ).astype(int)

    print(f"  - is_international: {df['is_international'].sum()} 国际选手, "
          f"{(~df['is_international'].astype(bool)).sum()} 美国选手")

    return df


def add_state_population_features(df, config):
    """
    添加州人口特征

    特征：
    - log_us_state_pop: 美国选手所在州的log10(人口数)，国际选手填0
    """
    # 读取州人口数据
    pop_file = config.get('state_population_file', 'NST-EST2025-POP.xlsx')

    try:
        pop_df = pd.read_excel(pop_file)

        # 清理数据：提取州名和average列
        # 第一列包含州名（带点号前缀），最后一列是average
        pop_df.columns = ['state'] + [f'col{i}' for i in range(1, len(pop_df.columns))]
        pop_df = pop_df.iloc[1:]  # 跳过第一行（列名行）

        # 提取州名（去掉点号前缀）
        pop_df['state'] = pop_df['state'].str.replace('.', '', regex=False).str.strip()

        # 提取average列（最后一列）
        pop_df['population'] = pop_df.iloc[:, -1]

        # 转换为数值
        pop_df['population'] = pd.to_numeric(pop_df['population'], errors='coerce')

        # 创建州名到人口的映射
        state_pop_dict = dict(zip(pop_df['state'], pop_df['population']))

        print(f"  - 成功加载 {len(state_pop_dict)} 个州的人口数据")

        # 州名标准化映射（处理数据中的特殊情况和拼写错误）
        state_name_corrections = {
            'Washington D.C.': 'Washington',  # D.C.在人口表中为Washington
            'New Hamshire': 'New Hampshire'   # 拼写错误修正
        }

        # 计算log_us_state_pop
        def get_log_pop(row):
            if row['is_international'] == 1:
                return 0
            else:
                state = row['celebrity_homestate']
                if pd.isna(state):
                    return 0

                # 应用州名修正
                state = state_name_corrections.get(state, state)

                pop = state_pop_dict.get(state, None)
                if pop is None or pd.isna(pop):
                    print(f"    警告：未找到州 '{state}' 的人口数据")
                    return 0
                return np.log10(pop)

        df['log_us_state_pop'] = df.apply(get_log_pop, axis=1)

        print(f"  - log_us_state_pop: 均值={df[df['is_international']==0]['log_us_state_pop'].mean():.2f}")

    except Exception as e:
        print(f"  - 错误：无法加载州人口数据: {e}")
        df['log_us_state_pop'] = 0

    return df
"""
职业舞者特征模块

构建职业舞者相关特征
"""

import pandas as pd
import numpy as np


def add_pro_features(config, datas):
    """
    添加职业舞者特征

    包括：
    - pro_id: 舞者唯一ID
    - partner_previous_seasons: 本赛季之前的参赛次数（已在宽格式中计算）
    - pro_prev_wins: 本赛季之前的冠军次数
    - pro_avg_rank: 本赛季之前的历史平均排名
    - same_sex_pair: 同性配对标志

    Args:
        config: 配置字典
        datas: 数据字典，包含 'long_data' 键

    Returns:
        datas: 更新后的数据字典
    """
    long_df = datas['long_data'].copy()

    print("=" * 60)
    print("添加职业舞者特征")
    print("=" * 60)

    # 1. 分配pro_id
    print("\n[1/4] 分配职业舞者ID...")
    long_df, pro_id_map = assign_pro_ids(long_df)

    # 2. 验证参赛次数特征已存在
    print("\n[2/4] 验证职业舞者历史参赛次数...")
    if 'partner_previous_seasons' in long_df.columns:
        max_seasons = int(long_df['partner_previous_seasons'].max())
        print(f"  - 职业舞者历史参赛次数已从宽格式携带（最多 {max_seasons} 季）")
    else:
        print("  - 警告：未找到partner_previous_seasons特征")

    # 3. 计算历史冠军次数
    print("\n[3/4] 计算职业舞者历史冠军次数...")
    long_df = calculate_pro_prev_wins(long_df, datas)

    # 4. 计算历史平均排名
    print("\n[4/4] 计算职业舞者历史平均排名...")
    long_df = calculate_pro_avg_rank(long_df, datas)

    # 5. 添加同性配对标志
    print("\n[5/5] 添加同性配对标志...")
    long_df = add_same_sex_pair(long_df, datas)

    print("\n" + "=" * 60)
    print("职业舞者特征添加完成")
    print("=" * 60)

    datas['long_data'] = long_df
    datas['pro_id_map'] = pro_id_map
    return datas


def assign_pro_ids(long_df):
    """
    为每位职业舞者分配唯一ID

    Returns:
        long_df: 更新后的数据框
        pro_id_map: 舞者名字到ID的映射
    """
    # 获取所有唯一的舞者名字
    unique_pros = long_df['ballroom_partner'].unique()

    # 分配ID（从1开始）
    pro_id_map = {name: idx + 1 for idx, name in enumerate(sorted(unique_pros))}

    # 添加pro_id列
    long_df['pro_id'] = long_df['ballroom_partner'].map(pro_id_map)

    print(f"  - 共有 {len(pro_id_map)} 位职业舞者")

    return long_df, pro_id_map


def calculate_pro_prev_wins(long_df, datas):
    """
    计算职业舞者在本赛季之前的冠军次数

    注意：严格遵守时间序列约束
    """
    # 从原始数据中获取每个赛季的冠军信息
    raw_df = datas['raw_data']

    # 找出每个赛季的冠军及其舞者
    champions = raw_df[raw_df['results'] == '1st Place'][['season', 'ballroom_partner']]

    # 为每个舞者在每个赛季计算之前的冠军次数
    pro_wins_dict = {}

    for season in sorted(long_df['season_id'].unique()):
        # 获取该赛季之前的所有冠军
        prev_champions = champions[champions['season'] < season]

        # 统计每个舞者的冠军次数
        wins_count = prev_champions['ballroom_partner'].value_counts().to_dict()

        # 存储
        pro_wins_dict[season] = wins_count

    # 应用到长格式数据
    def get_prev_wins(row):
        season = row['season_id']
        partner = row['ballroom_partner']

        if season not in pro_wins_dict:
            return 0

        return pro_wins_dict[season].get(partner, 0)

    long_df['pro_prev_wins'] = long_df.apply(get_prev_wins, axis=1)

    max_wins = int(long_df['pro_prev_wins'].max())
    print(f"  - 职业舞者历史冠军次数计算完成（最多 {max_wins} 次）")

    return long_df


def calculate_pro_avg_rank(long_df, datas):
    """
    计算职业舞者在本赛季之前的历史平均排名

    若为新舞者，填所有舞者的平均值
    """
    raw_df = datas['raw_data']

    # 为每个舞者在每个赛季计算之前的平均排名
    pro_rank_dict = {}

    for season in sorted(long_df['season_id'].unique()):
        # 获取该赛季之前的所有记录
        prev_records = raw_df[raw_df['season'] < season]

        # 计算每个舞者的平均排名
        pro_ranks = prev_records.groupby('ballroom_partner')['placement'].mean().to_dict()

        # 存储
        pro_rank_dict[season] = pro_ranks

    # 计算全局平均排名（用于新舞者）
    global_avg_rank = raw_df['placement'].mean()

    # 应用到长格式数据
    def get_avg_rank(row):
        season = row['season_id']
        partner = row['ballroom_partner']

        if season not in pro_rank_dict:
            return global_avg_rank

        return pro_rank_dict[season].get(partner, global_avg_rank)

    long_df['pro_avg_rank'] = long_df.apply(get_avg_rank, axis=1)

    print(f"  - pro_avg_rank: 均值={long_df['pro_avg_rank'].mean():.2f}, "
          f"全局平均={global_avg_rank:.2f}")

    return long_df


def add_same_sex_pair(long_df, datas):
    """
    添加同性配对标志

    直接从长格式数据中读取性别信息
    """
    # 检查性别特征是否存在
    if 'celebrity_gender' not in long_df.columns or 'ballroom_partner_gender' not in long_df.columns:
        print("  - 警告：性别特征不存在，跳过same_sex_pair")
        long_df['same_sex_pair'] = 0
        return long_df

    # 应用到长格式数据
    def is_same_sex(row):
        celebrity_gender = row['celebrity_gender']
        partner_gender = row['ballroom_partner_gender']

        # 如果任一性别未知，返回0
        if celebrity_gender == 'unknown' or partner_gender == 'unknown':
            return 0

        return 1 if celebrity_gender == partner_gender else 0

    long_df['same_sex_pair'] = long_df.apply(is_same_sex, axis=1)

    print(f"  - same_sex_pair: {long_df['same_sex_pair'].sum()} 个同性配对")

    return long_df
"""
动态特征模块

构建每周动态变化的特征
"""

import pandas as pd
import numpy as np


def add_dynamic_features(config, datas):
    """
    添加动态表现特征

    包括：
    - z_score: 标准化评委分数
    - score_trend: 本周与上周z_score的差值
    - is_top_score: 是否当周最高分
    - perfect_score: 是否满分
    - judge_score_stddev: 评委评分标准差

    Args:
        config: 配置字典
        datas: 数据字典，包含 'long_data' 键

    Returns:
        datas: 更新后的数据字典
    """
    long_df = datas['long_data'].copy()
    missing_judge4_weeks = datas.get('missing_judge4_weeks', [])

    print("=" * 60)
    print("添加动态特征")
    print("=" * 60)

    # 1. 计算原始评委分数
    print("\n[1/5] 计算原始评委分数...")
    long_df = calculate_raw_scores(long_df, missing_judge4_weeks)

    # 2. 计算z_score
    print("\n[2/5] 计算z_score...")
    long_df = calculate_z_score(long_df)

    # 3. 计算score_trend
    print("\n[3/5] 计算score_trend...")
    long_df = calculate_score_trend(long_df)

    # 4. 标记最高分和满分
    print("\n[4/5] 标记最高分和满分...")
    long_df = mark_top_and_perfect_scores(long_df, missing_judge4_weeks)

    # 5. 计算评委评分标准差
    print("\n[5/5] 计算评委评分标准差...")
    long_df = calculate_judge_stddev(long_df)

    print("\n" + "=" * 60)
    print("动态特征添加完成")
    print("=" * 60)

    datas['long_data'] = long_df
    return datas


def calculate_raw_scores(long_df, missing_judge4_weeks):
    """
    计算原始评委总分

    考虑3个或4个评委的情况
    """
    def get_raw_score(row):
        season = row['season_id']
        week = row['week_id']

        # 检查该周是否只有3个评委
        has_judge4 = (season, week) not in missing_judge4_weeks

        scores = []
        for j in range(1, 5 if has_judge4 else 4):
            score = row.get(f'judge{j}_score', np.nan)
            if pd.notna(score):
                scores.append(score)

        if len(scores) == 0:
            return 0  # 如果没有有效分数，返回0而不是NaN

        return np.sum(scores)

    long_df['judge_score_raw'] = long_df.apply(get_raw_score, axis=1)

    print(f"  - 原始评委分数计算完成")

    return long_df


def calculate_z_score(long_df):
    """
    计算z_score（标准化评委分数）

    公式：(个人分 - 当周全员均分) / 当周标准差
    """
    # 按赛季和周分组计算z_score
    def calc_z(group):
        mean_score = group['judge_score_raw'].mean()
        std_score = group['judge_score_raw'].std()

        if std_score == 0 or pd.isna(std_score):
            group = group.copy()
            group['z_score'] = 0
        else:
            group = group.copy()
            group['z_score'] = (group['judge_score_raw'] - mean_score) / std_score

        return group

    long_df = long_df.groupby(['season_id', 'week_id'], group_keys=False).apply(calc_z)

    print(f"  - 标准化评委分数计算完成")

    return long_df


def calculate_score_trend(long_df):
    """
    计算score_trend（本周与上周z_score的差值）

    第一周填0
    """
    # 按选手和赛季分组，计算上周的z_score
    long_df = long_df.sort_values(['season_id', 'celebrity_name', 'week_id'])

    def calc_trend(group):
        group = group.copy()
        group['prev_z_score'] = group['z_score'].shift(1)
        group['prev_z_score'] = group['prev_z_score'].fillna(0)  # 第一周填0
        group['score_trend'] = group['z_score'] - group['prev_z_score']
        group['score_trend'] = group['score_trend'].fillna(0)  # 第一周填0
        return group

    long_df = long_df.groupby(['season_id', 'celebrity_name'], group_keys=False).apply(calc_trend)

    print(f"  - 分数趋势计算完成")

    return long_df


def mark_top_and_perfect_scores(long_df, missing_judge4_weeks):
    """
    标记最高分和满分

    - is_top_score: 当周最高分（或并列最高）
    - perfect_score: 满分（3个评委30分，4个评委40分）
    """
    # 按赛季和周分组
    def mark_top(group):
        group = group.copy()
        max_score = group['judge_score_raw'].max()
        group['is_top_score'] = (group['judge_score_raw'] == max_score).astype(int)
        return group

    long_df = long_df.groupby(['season_id', 'week_id'], group_keys=False).apply(mark_top)

    # 标记满分
    def is_perfect(row):
        season = row['season_id']
        week = row['week_id']
        score = row['judge_score_raw']

        # 检查该周是否只有3个评委
        has_judge4 = (season, week) not in missing_judge4_weeks

        perfect_threshold = 40 if has_judge4 else 30

        return 1 if score >= perfect_threshold else 0

    long_df['perfect_score'] = long_df.apply(is_perfect, axis=1)

    print(f"  - is_top_score: {long_df['is_top_score'].sum()} 次最高分")
    print(f"  - perfect_score: {long_df['perfect_score'].sum()} 次满分")

    return long_df


def calculate_judge_stddev(long_df):
    """
    计算评委评分标准差

    量化评委间的"争议度"
    """
    def get_stddev(row):
        scores = []
        for j in range(1, 5):
            score = row.get(f'judge{j}_score', np.nan)
            if pd.notna(score) and score != 0:
                scores.append(score)

        if len(scores) < 2:
            return 0

        return np.std(scores)

    long_df['judge_score_stddev'] = long_df.apply(get_stddev, axis=1)

    print(f"  - 评委评分标准差计算完成")

    return long_df
"""
历史轨迹特征模块

构建选手的历史轨迹特征
"""

import pandas as pd
import numpy as np


def add_survival_features(config, datas):
    """
    添加存活与历史轨迹特征

    包括：
    - times_in_bottom: 累计进入倒数两名的次数
    - cumulative_avg_score: 赛季累计平均分
    - teflon_factor: 上周倒数前2但本周安全
    - weeks_survived: 当前周数

    Args:
        config: 配置字典
        datas: 数据字典，包含 'long_data' 键

    Returns:
        datas: 更新后的数据字典
    """
    long_df = datas['long_data'].copy()

    print("=" * 60)
    print("添加历史轨迹特征")
    print("=" * 60)

    # 1. 计算每周的排名
    print("\n[1/4] 计算每周评委分数排名...")
    long_df = calculate_weekly_rank(long_df)

    # 2. 计算times_in_bottom
    print("\n[2/4] 计算累计倒数次数...")
    long_df = calculate_times_in_bottom(long_df)

    # 3. 计算cumulative_avg_score
    print("\n[3/4] 计算累计平均分...")
    long_df = calculate_cumulative_avg_score(long_df)

    # 4. 计算teflon_factor
    print("\n[4/4] 计算teflon_factor...")
    long_df = calculate_teflon_factor(long_df)

    # 5. 添加weeks_survived
    long_df['weeks_survived'] = long_df['week_id']

    print("\n" + "=" * 60)
    print("历史轨迹特征添加完成")
    print("=" * 60)

    datas['long_data'] = long_df
    return datas


def calculate_weekly_rank(long_df):
    """
    计算每周的评委分数排名

    1 = 最高分，数字越大排名越低
    """
    # 按赛季和周分组，计算排名
    def calc_rank(group):
        group = group.copy()
        # 排名：分数越高排名越靠前（rank=1是最高分）
        group['judge_rank'] = group['judge_score_raw'].rank(ascending=False, method='min')
        return group

    long_df = long_df.groupby(['season_id', 'week_id'], group_keys=False).apply(calc_rank)

    print(f"  - 每周排名计算完成")

    return long_df


def calculate_times_in_bottom(long_df):
    """
    计算累计进入倒数两名的次数

    注意：
    - 每周都判定倒数两名（基于评委分数排名）
    - 截止到本周前的累计次数（不包括本周）
    """
    # 按选手和赛季分组
    long_df = long_df.sort_values(['season_id', 'celebrity_name', 'week_id'])

    def calc_bottom_times(group):
        group = group.copy()
        # 判定每周是否在倒数两名
        # 需要知道该周有多少选手
        group['is_bottom_two'] = 0

        for idx in group.index:
            season = group.loc[idx, 'season_id']
            week = group.loc[idx, 'week_id']
            rank = group.loc[idx, 'judge_rank']

            # 获取该周的总选手数
            week_data = long_df[(long_df['season_id'] == season) &
                               (long_df['week_id'] == week)]
            total_contestants = len(week_data)

            # 倒数两名：排名在倒数第1和倒数第2
            # 即 rank >= total_contestants - 1
            if rank >= total_contestants - 1:
                group.loc[idx, 'is_bottom_two'] = 1

        # 计算累计次数（截止到本周前）
        group['times_in_bottom'] = group['is_bottom_two'].shift(1).fillna(0).cumsum()

        return group

    long_df = long_df.groupby(['season_id', 'celebrity_name'], group_keys=False).apply(calc_bottom_times)

    print(f"  - 累计倒数次数计算完成")

    return long_df


def calculate_cumulative_avg_score(long_df):
    """
    计算赛季累计平均分

    截止到本周前的平均分（不包括本周）
    """
    long_df = long_df.sort_values(['season_id', 'celebrity_name', 'week_id'])

    def calc_cumulative_avg(group):
        group = group.copy()
        # 计算累计平均分（截止到本周前）
        cumsum = group['judge_score_raw'].shift(1).fillna(0).cumsum()

        # 使用组内相对位置作为计数，而不是绝对周数
        # 这样可以正确处理中途加入的选手
        count = np.arange(len(group))  # 0, 1, 2, 3, ...
        count = np.where(count == 0, 1, count)  # 避免除以0

        group['cumulative_avg_score'] = cumsum / count

        # 第一次出现（组内第一周）填0
        group['cumulative_avg_score'].iloc[0] = 0

        return group

    long_df = long_df.groupby(['season_id', 'celebrity_name'], group_keys=False).apply(calc_cumulative_avg)

    print(f"  - 累计平均分计算完成")

    return long_df


def calculate_teflon_factor(long_df):
    """
    计算teflon_factor

    定义：上一周评委分排倒数前2，但本周结果为Safe（未被淘汰）

    这是最强特征：说明该选手粉丝票极高
    """
    long_df = long_df.sort_values(['season_id', 'celebrity_name', 'week_id'])

    def calc_teflon(group):
        group = group.copy()
        group['teflon_factor'] = 0

        for i in range(1, len(group)):
            prev_idx = group.index[i - 1]
            curr_idx = group.index[i]

            # 上周是否在倒数前2
            prev_is_bottom = group.loc[prev_idx, 'is_bottom_two']

            # 本周是否安全（未被淘汰）
            curr_status = group.loc[curr_idx, 'result_status']
            curr_is_safe = (curr_status == 0)

            # 如果上周倒数但本周安全
            if prev_is_bottom == 1 and curr_is_safe:
                group.loc[curr_idx, 'teflon_factor'] = 1

        return group

    long_df = long_df.groupby(['season_id', 'celebrity_name'], group_keys=False).apply(calc_teflon)

    teflon_count = long_df['teflon_factor'].sum()
    print(f"  - 特氟龙效应（Teflon Factor）: 共 {int(teflon_count)} 次")

    return long_df
"""
赛制特征模块

构建赛制和环境特征
"""

import pandas as pd
import numpy as np


def add_context_features(config, datas):
    """
    添加赛制与环境特征

    包括：
    - season_era: 赛季时代（1=早期, 2=中期, 3=社媒期）
    - rule_method: 投票规则（0=排名法, 1=百分比法）
    - judge_save_active: 评委拯救机制（0/1）
    - elimination_count: 本周淘汰人数

    Args:
        config: 配置字典
        datas: 数据字典，包含 'long_data' 键

    Returns:
        datas: 更新后的数据字典
    """
    long_df = datas['long_data'].copy()

    print("=" * 60)
    print("添加赛制特征")
    print("=" * 60)

    # 1. 赛季时代
    print("\n[1/4] 添加赛季时代...")
    long_df = add_season_era(long_df)

    # 2. 投票规则
    print("\n[2/4] 添加投票规则...")
    long_df = add_voting_rule(long_df)

    # 3. 评委拯救机制
    print("\n[3/4] 添加评委拯救机制...")
    long_df = add_judge_save(long_df)

    # 4. 每周淘汰人数
    print("\n[4/4] 添加每周淘汰人数...")
    long_df = add_weekly_eliminations(long_df)

    print("\n" + "=" * 60)
    print("赛制特征添加完成")
    print("=" * 60)

    datas['long_data'] = long_df
    return datas


def add_season_era(long_df):
    """
    添加赛季时代

    1: 早期(S1-10)
    2: 中期(S11-20)
    3: 社媒期(S21+)
    """
    def get_era(season):
        if season <= 10:
            return 1
        elif season <= 20:
            return 2
        else:
            return 3

    long_df['season_era'] = long_df['season_id'].apply(get_era)

    print(f"  - 早期(S1-10): {(long_df['season_era'] == 1).sum()} 条记录")
    print(f"  - 中期(S11-20): {(long_df['season_era'] == 2).sum()} 条记录")
    print(f"  - 社媒期(S21+): {(long_df['season_era'] == 3).sum()} 条记录")

    return long_df


def add_voting_rule(long_df):
    """
    添加投票规则

    0 = 排名法 (Rank Method): S1-2, S28+
    1 = 百分比法 (Percent Method): S3-27
    """
    def get_rule(season):
        if season <= 2:
            return 0  # 排名法
        elif season <= 27:
            return 1  # 百分比法
        else:
            return 0  # 排名法

    long_df['voting_rule_type'] = long_df['season_id'].apply(get_rule)

    print(f"  - 排名法: {(long_df['voting_rule_type'] == 0).sum()} 条记录")
    print(f"  - 百分比法: {(long_df['voting_rule_type'] == 1).sum()} 条记录")

    return long_df


def add_judge_save(long_df):
    """
    添加评委拯救机制

    S28+ 有评委拯救机制
    """
    long_df['judge_save_active'] = (long_df['season_id'] >= 28).astype(int)

    print(f"  - 有评委拯救: {long_df['judge_save_active'].sum()} 条记录")
    print(f"  - 无评委拯救: {(~long_df['judge_save_active'].astype(bool)).sum()} 条记录")

    return long_df


def add_weekly_eliminations(long_df):
    """
    添加每周淘汰人数特征

    统计每个赛季每周淘汰了多少人
    """
    # 按赛季和周分组，统计淘汰人数
    elimination_counts = long_df[long_df['result_status'] == 1].groupby(
        ['season_id', 'week_id']
    ).size().to_dict()

    # 应用到所有记录
    def get_eliminations(row):
        key = (row['season_id'], row['week_id'])
        return elimination_counts.get(key, 0)

    long_df['elimination_count'] = long_df.apply(get_eliminations, axis=1)

    # 统计信息
    unique_eliminations = long_df.groupby(['season_id', 'week_id'])['elimination_count'].first()

    print(f"  - 无淘汰周: {(unique_eliminations == 0).sum()} 周")
    print(f"  - 单淘汰周: {(unique_eliminations == 1).sum()} 周")
    print(f"  - 双淘汰周: {(unique_eliminations == 2).sum()} 周")
    if (unique_eliminations > 2).any():
        print(f"  - 三人及以上淘汰周: {(unique_eliminations > 2).sum()} 周")
    print(f"  - 最多淘汰人数: {int(unique_eliminations.max())} 人")

    return long_df
