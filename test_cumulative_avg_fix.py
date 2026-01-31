"""
验证修复后的 cumulative_avg_score 计算逻辑
"""

import pandas as pd
import numpy as np

print("=" * 60)
print("测试修复后的 cumulative_avg_score 计算逻辑")
print("=" * 60)

# 创建测试数据
test_cases = [
    {
        'name': '从第1周开始的选手',
        'data': pd.DataFrame({
            'season_id': [1, 1, 1, 1],
            'celebrity_name': ['Alice', 'Alice', 'Alice', 'Alice'],
            'week_id': [1, 2, 3, 4],
            'judge_score_raw': [24, 27, 30, 28]
        }),
        'expected': [0, 24.0, 25.5, 27.0]
    },
    {
        'name': '从第3周开始的选手（中途加入）',
        'data': pd.DataFrame({
            'season_id': [1, 1, 1],
            'celebrity_name': ['Bob', 'Bob', 'Bob'],
            'week_id': [3, 4, 5],
            'judge_score_raw': [20, 25, 30]
        }),
        'expected': [0, 20.0, 22.5]
    },
    {
        'name': '从第5周开始的选手',
        'data': pd.DataFrame({
            'season_id': [2, 2, 2, 2],
            'celebrity_name': ['Charlie', 'Charlie', 'Charlie', 'Charlie'],
            'week_id': [5, 6, 7, 8],
            'judge_score_raw': [18, 22, 26, 30]
        }),
        'expected': [0, 18.0, 20.0, 22.0]
    }
]

# 修复后的计算函数
def calc_cumulative_avg_fixed(group):
    group = group.copy()
    # 计算累计平均分（截止到本周前）
    cumsum = group['judge_score_raw'].shift(1).fillna(0).cumsum()

    # 使用组内相对位置作为计数，而不是绝对周数
    count = np.arange(len(group))  # 0, 1, 2, 3, ...
    count = np.where(count == 0, 1, count)  # 避免除以0

    group['cumulative_avg_score'] = cumsum / count

    # 第一次出现（组内第一周）填0
    group['cumulative_avg_score'].iloc[0] = 0

    return group

# 旧的计算函数（有bug）
def calc_cumulative_avg_old(group):
    group = group.copy()
    cumsum = group['judge_score_raw'].shift(1).fillna(0).cumsum()
    count = group['week_id'] - 1
    count = count.replace(0, 1)

    group['cumulative_avg_score'] = cumsum / count
    group.loc[group['week_id'] == 1, 'cumulative_avg_score'] = 0

    return group

# 测试每个案例
all_passed = True

for i, test_case in enumerate(test_cases, 1):
    print(f"\n{'=' * 60}")
    print(f"测试案例 {i}: {test_case['name']}")
    print(f"{'=' * 60}")

    df = test_case['data'].copy()
    expected = test_case['expected']

    # 应用修复后的函数
    df_fixed = df.groupby(['season_id', 'celebrity_name'], group_keys=False).apply(calc_cumulative_avg_fixed)

    # 应用旧的函数（对比）
    df_old = df.copy()
    df_old = df_old.groupby(['season_id', 'celebrity_name'], group_keys=False).apply(calc_cumulative_avg_old)

    print(f"\n{'Week':<6} {'Raw Score':<12} {'Expected':<12} {'Fixed':<12} {'Old (Bug)':<12} {'Status':<10}")
    print("-" * 70)

    for idx, row in df_fixed.iterrows():
        week = row['week_id']
        raw_score = row['judge_score_raw']
        expected_val = expected[idx]
        fixed_val = row['cumulative_avg_score']
        old_val = df_old.loc[idx, 'cumulative_avg_score']

        # 检查修复后的值是否正确
        is_correct = abs(fixed_val - expected_val) < 0.01
        status = "✓ PASS" if is_correct else "✗ FAIL"

        if not is_correct:
            all_passed = False

        print(f"{week:<6} {raw_score:<12.2f} {expected_val:<12.2f} {fixed_val:<12.2f} {old_val:<12.2f} {status:<10}")

    # 显示差异
    if test_case['name'] != '从第1周开始的选手':
        print(f"\n分析:")
        print(f"  - 该选手从 Week {df['week_id'].iloc[0]} 开始")
        print(f"  - 旧逻辑使用 'week_id - 1' 作为计数，导致分母过大")
        print(f"  - 新逻辑使用组内相对位置，正确处理中途加入的情况")

print("\n" + "=" * 60)
print("测试总结")
print("=" * 60)

if all_passed:
    print("\n✓ 所有测试通过！修复后的逻辑正确")
else:
    print("\n✗ 部分测试失败，需要进一步检查")

print("\n修复说明:")
print("  1. 将 'count = group['week_id'] - 1' 改为 'count = np.arange(len(group))'")
print("  2. 使用组内相对位置而不是绝对周数")
print("  3. 正确处理中途加入的选手")
print("  4. 第一次出现时累计平均分为0")
