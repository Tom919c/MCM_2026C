"""
测试脚本：打印 train_data 的完整结构
"""

import pickle
import numpy as np


def print_train_data_structure(datas):
    """
    打印 train_data 的完整结构

    Args:
        datas: 数据字典
    """
    print("=" * 80)
    print("train_data 结构详情")
    print("=" * 80)

    train_data = datas['train_data']

    # ========== 1. 维度信息 ==========
    print("\n[1] 维度信息")
    print("-" * 80)
    print(f"  n_obs (观测总数):        {train_data['n_obs']}")
    print(f"  n_weeks (周总数):        {train_data['n_weeks']}")
    print(f"  n_celebs (名人总数):     {train_data['n_celebs']}")
    print(f"  n_pros (职业舞者总数):   {train_data['n_pros']}")
    print(f"  n_seasons (赛季总数):    {train_data['n_seasons']}")

    # ========== 2. 索引数组 ==========
    print("\n[2] 索引数组")
    print("-" * 80)
    print(f"  celeb_idx:   shape={train_data['celeb_idx'].shape}, "
          f"dtype={train_data['celeb_idx'].dtype}, "
          f"range=[{train_data['celeb_idx'].min()}, {train_data['celeb_idx'].max()}]")
    print(f"  pro_idx:     shape={train_data['pro_idx'].shape}, "
          f"dtype={train_data['pro_idx'].dtype}, "
          f"range=[{train_data['pro_idx'].min()}, {train_data['pro_idx'].max()}]")
    print(f"  week_idx:    shape={train_data['week_idx'].shape}, "
          f"dtype={train_data['week_idx'].dtype}, "
          f"range=[{train_data['week_idx'].min()}, {train_data['week_idx'].max()}]")
    print(f"  season_idx:  shape={train_data['season_idx'].shape}, "
          f"dtype={train_data['season_idx'].dtype}, "
          f"range=[{train_data['season_idx'].min()}, {train_data['season_idx'].max()}]")

    # ========== 3. 特征矩阵 ==========
    print("\n[3] 特征矩阵")
    print("-" * 80)

    # X_celeb
    print(f"\n  X_celeb:")
    print(f"    shape: {train_data['X_celeb'].shape}")
    print(f"    dtype: {train_data['X_celeb'].dtype}")
    print(f"    特征数: {len(train_data['X_celeb_names'])}")
    print(f"    特征列表:")
    for i, name in enumerate(train_data['X_celeb_names']):
        col = train_data['X_celeb'][:, i]
        print(f"      [{i:2d}] {name:50s} | "
              f"mean={np.nanmean(col):7.3f}, std={np.nanstd(col):7.3f}, "
              f"min={np.nanmin(col):7.3f}, max={np.nanmax(col):7.3f}")

    # X_pro
    print(f"\n  X_pro:")
    print(f"    shape: {train_data['X_pro'].shape}")
    print(f"    dtype: {train_data['X_pro'].dtype}")
    print(f"    特征数: {len(train_data['X_pro_names'])}")
    print(f"    特征列表:")
    for i, name in enumerate(train_data['X_pro_names']):
        col = train_data['X_pro'][:, i]
        print(f"      [{i:2d}] {name:50s} | "
              f"mean={np.nanmean(col):7.3f}, std={np.nanstd(col):7.3f}, "
              f"min={np.nanmin(col):7.3f}, max={np.nanmax(col):7.3f}")

    # X_obs
    print(f"\n  X_obs:")
    print(f"    shape: {train_data['X_obs'].shape}")
    print(f"    dtype: {train_data['X_obs'].dtype}")
    print(f"    特征数: {len(train_data['X_obs_names'])}")
    print(f"    特征列表:")
    for i, name in enumerate(train_data['X_obs_names']):
        col = train_data['X_obs'][:, i]
        print(f"      [{i:2d}] {name:50s} | "
              f"mean={np.nanmean(col):7.3f}, std={np.nanstd(col):7.3f}, "
              f"min={np.nanmin(col):7.3f}, max={np.nanmax(col):7.3f}")

    # ========== 4. 淘汰规则相关数据 ==========
    print("\n[4] 淘汰规则相关数据")
    print("-" * 80)
    print(f"  judge_score_pct:")
    print(f"    shape: {train_data['judge_score_pct'].shape}")
    print(f"    dtype: {train_data['judge_score_pct'].dtype}")
    print(f"    range: [{train_data['judge_score_pct'].min():.3f}, "
          f"{train_data['judge_score_pct'].max():.3f}]")
    print(f"    mean: {train_data['judge_score_pct'].mean():.3f}")

    print(f"\n  judge_rank_score:")
    print(f"    shape: {train_data['judge_rank_score'].shape}")
    print(f"    dtype: {train_data['judge_rank_score'].dtype}")
    print(f"    range: [{train_data['judge_rank_score'].min():.3f}, "
          f"{train_data['judge_rank_score'].max():.3f}]")
    print(f"    mean: {train_data['judge_rank_score'].mean():.3f}")

    # ========== 5. 周级结构信息 ==========
    print("\n[5] 周级结构信息")
    print("-" * 80)
    print(f"  week_data 长度: {len(train_data['week_data'])}")

    if len(train_data['week_data']) > 0:
        # 显示第一周的详细信息
        week0 = train_data['week_data'][0]
        print(f"\n  示例：第0周数据结构")
        print(f"    键: {list(week0.keys())}")
        print(f"    obs_mask:        shape={week0['obs_mask'].shape}, "
              f"dtype={week0['obs_mask'].dtype}, sum={week0['obs_mask'].sum()}")
        print(f"    n_contestants:   {week0['n_contestants']}")
        print(f"    n_eliminated:    {week0['n_eliminated']}")
        print(f"    eliminated_mask: shape={week0['eliminated_mask'].shape}, "
              f"sum={week0['eliminated_mask'].sum()}")
        print(f"    rule_method:     {week0['rule_method']} "
              f"(0=排名法, 1=百分比法)")
        print(f"    judge_save_active: {week0['judge_save_active']}")
        print(f"    season:          {week0['season']}")
        print(f"    week:            {week0['week']}")

        # 统计信息
        print(f"\n  周级数据统计:")
        total_contestants = sum(w['n_contestants'] for w in train_data['week_data'])
        total_eliminated = sum(w['n_eliminated'] for w in train_data['week_data'])
        rule_method_counts = {}
        for w in train_data['week_data']:
            rm = w['rule_method']
            rule_method_counts[rm] = rule_method_counts.get(rm, 0) + 1

        print(f"    总参赛人次: {total_contestants}")
        print(f"    总淘汰人次: {total_eliminated}")
        print(f"    投票规则分布: {rule_method_counts}")

    # ========== 6. 数据完整性检查 ==========
    print("\n[6] 数据完整性检查")
    print("-" * 80)

    # 检查 NaN 值
    celeb_nan = np.isnan(train_data['X_celeb']).sum()
    pro_nan = np.isnan(train_data['X_pro']).sum()
    obs_nan = np.isnan(train_data['X_obs']).sum()

    print(f"  X_celeb NaN 数量: {celeb_nan}")
    print(f"  X_pro NaN 数量:   {pro_nan}")
    print(f"  X_obs NaN 数量:   {obs_nan}")

    # 检查索引范围
    print(f"\n  索引范围检查:")
    print(f"    celeb_idx 是否在 [0, {train_data['n_celebs']-1}]: "
          f"{train_data['celeb_idx'].min() >= 0 and train_data['celeb_idx'].max() < train_data['n_celebs']}")
    print(f"    pro_idx 是否在 [0, {train_data['n_pros']-1}]: "
          f"{train_data['pro_idx'].min() >= 0 and train_data['pro_idx'].max() < train_data['n_pros']}")
    print(f"    week_idx 是否在 [0, {train_data['n_weeks']-1}]: "
          f"{train_data['week_idx'].min() >= 0 and train_data['week_idx'].max() < train_data['n_weeks']}")

    print("\n" + "=" * 80)
    print("结构打印完成")
    print("=" * 80)


if __name__ == "__main__":
    # 加载数据
    print("加载数据...")
    try:
        with open('data/processed/datas.pkl', 'rb') as f:
            datas = pickle.load(f)
        print("数据加载成功\n")

        # 打印结构
        print_train_data_structure(datas)

    except FileNotFoundError:
        print("错误: 找不到文件 'data/processed/datas.pkl'")
        print("请先运行数据处理流程生成该文件")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
