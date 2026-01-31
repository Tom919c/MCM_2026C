"""
检查名人是否在同一赛季更换搭档
"""

import pickle
import pandas as pd

print("=" * 60)
print("加载数据...")
print("=" * 60)

with open('data/processed/datas.pkl', 'rb') as f:
    datas = pickle.load(f)

if 'long_data' not in datas:
    print("错误: 没有找到 long_data")
    exit(1)

long_data = datas['long_data']
print(f"long_data 形状: {long_data.shape}")

# 检查每个名人在每个赛季有多少个不同的搭档
print("\n" + "=" * 60)
print("检查名人在同一赛季的搭档数量")
print("=" * 60)

# 按赛季和名人分组，统计不同搭档的数量
partner_counts = long_data.groupby(['season_id', 'celebrity_name'])['ballroom_partner'].nunique()

# 找出有多个搭档的情况
multiple_partners = partner_counts[partner_counts > 1]

print(f"\n总组合数（按 season + celebrity）: {len(partner_counts)}")
print(f"单一搭档: {(partner_counts == 1).sum()}")
print(f"多个搭档: {len(multiple_partners)}")

if len(multiple_partners) > 0:
    print("\n" + "=" * 60)
    print("名人在同一赛季更换搭档的情况")
    print("=" * 60)

    print(f"\n共 {len(multiple_partners)} 个名人在同一赛季有多个搭档:")
    print(f"\n{'Season':<8} {'Celebrity Name':<30} {'Partner Count':<15}")
    print("-" * 60)

    for (season, name), count in multiple_partners.items():
        print(f"{season:<8} {name:<30} {count:<15}")

    # 详细查看第一个案例
    print("\n" + "=" * 60)
    print("详细案例分析")
    print("=" * 60)

    season, name = multiple_partners.index[0]
    print(f"\n名人: {name}")
    print(f"赛季: {season}")

    # 获取该名人在该赛季的所有记录
    contestant_data = long_data[
        (long_data['season_id'] == season) &
        (long_data['celebrity_name'] == name)
    ].sort_values('week_id')

    print(f"\n{'Week':<6} {'Ballroom Partner':<30} {'Judge Score':<12}")
    print("-" * 50)

    for idx, row in contestant_data.iterrows():
        week = row['week_id']
        partner = row['ballroom_partner']
        score = row.get('judge_score_raw', 'N/A')

        print(f"{week:<6} {partner:<30} {score}")

    print("\n分析:")
    print(f"  - 该名人在赛季 {season} 中有 {multiple_partners.iloc[0]} 个不同的搭档")
    print(f"  - 当前代码使用 groupby(['season_id', 'celebrity_name'])")
    print(f"  - 这会将不同搭档的数据混在一起计算累计特征")
    print(f"  - 正确的做法应该是 groupby(['season_id', 'celebrity_name', 'ballroom_partner'])")

# 检查当前特征计算的影响
print("\n" + "=" * 60)
print("检查当前特征计算逻辑")
print("=" * 60)

print("\n当前代码中使用的分组方式:")
print("  - cumulative_avg_score: groupby(['season_id', 'celebrity_name'])")
print("  - times_in_bottom: groupby(['season_id', 'celebrity_name'])")
print("  - teflon_factor: groupby(['season_id', 'celebrity_name'])")

if len(multiple_partners) > 0:
    print("\n✗ 问题:")
    print(f"  - 有 {len(multiple_partners)} 个名人在同一赛季更换了搭档")
    print("  - 当前分组方式会将不同组合的数据混在一起")
    print("  - 累计特征会跨越不同的组合计算，这是不正确的")
    print("\n✓ 建议:")
    print("  - 应该使用 groupby(['season_id', 'celebrity_name', 'ballroom_partner'])")
    print("  - 或者创建一个 pair_id 来唯一标识每个组合")
else:
    print("\n✓ 好消息:")
    print("  - 数据中没有名人在同一赛季更换搭档的情况")
    print("  - 当前分组方式在这个数据集上是正确的")

# 检查跨赛季的情况
print("\n" + "=" * 60)
print("检查名人跨赛季参赛情况")
print("=" * 60)

# 统计每个名人参加了多少个赛季
celebrity_seasons = long_data.groupby('celebrity_name')['season_id'].nunique()
multi_season = celebrity_seasons[celebrity_seasons > 1]

print(f"\n总名人数: {len(celebrity_seasons)}")
print(f"单赛季: {(celebrity_seasons == 1).sum()}")
print(f"多赛季: {len(multi_season)}")

if len(multi_season) > 0:
    print(f"\n参加多个赛季的名人 (前10个):")
    print(f"\n{'Celebrity Name':<30} {'Season Count':<15}")
    print("-" * 50)

    for name, count in multi_season.head(10).items():
        print(f"{name:<30} {count:<15}")

    print("\n注意:")
    print("  - 跨赛季的累计特征应该分别计算")
    print("  - 当前代码使用 groupby(['season_id', 'celebrity_name']) 是正确的")
    print("  - 每个赛季的累计特征是独立的")

print("\n" + "=" * 60)
print("检查完成")
print("=" * 60)
