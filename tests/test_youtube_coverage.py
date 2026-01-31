"""
YouTube Data Coverage Analysis
统计YouTube爬虫数据相对于原始数据集的覆盖率
"""

import pandas as pd
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def read_csv_with_encoding(file_path):
    """尝试多种编码读取CSV文件"""
    encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252', 'gbk']
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    # 最后使用utf-8并忽略错误
    return pd.read_csv(file_path, encoding='utf-8', errors='ignore')


def calculate_coverage_by_season(raw_data_path, youtube_data_path):
    """
    计算每个赛季的YouTube数据覆盖率

    Args:
        raw_data_path: 原始数据集路径
        youtube_data_path: YouTube爬虫数据路径

    Returns:
        coverage_df: 包含每个赛季覆盖率统计的DataFrame
    """
    # 读取原始数据集
    print("Loading raw data...")
    raw_df = read_csv_with_encoding(raw_data_path)
    print(f"Raw data loaded: {len(raw_df)} records")

    # 读取YouTube爬虫数据
    print("Loading YouTube data...")
    youtube_df = read_csv_with_encoding(youtube_data_path)
    print(f"YouTube data loaded: {len(youtube_df)} records")

    # 统计原始数据集中每个赛季的唯一名人数量
    raw_celebrities_by_season = raw_df.groupby('season')['celebrity_name'].nunique()
    print(f"\nTotal seasons in raw data: {len(raw_celebrities_by_season)}")

    # 统计YouTube数据中每个赛季的唯一名人数量
    youtube_celebrities_by_season = youtube_df.groupby('season')['celebrity_name'].nunique()
    print(f"Total seasons in YouTube data: {len(youtube_celebrities_by_season)}")

    # 创建覆盖率统计DataFrame
    coverage_data = []

    for season in sorted(raw_celebrities_by_season.index):
        total_celebrities = raw_celebrities_by_season[season]
        crawled_celebrities = youtube_celebrities_by_season.get(season, 0)
        coverage_rate = (crawled_celebrities / total_celebrities * 100) if total_celebrities > 0 else 0

        coverage_data.append({
            'season': season,
            'total_celebrities': total_celebrities,
            'crawled_celebrities': crawled_celebrities,
            'missing_celebrities': total_celebrities - crawled_celebrities,
            'coverage_rate': coverage_rate
        })

    coverage_df = pd.DataFrame(coverage_data)

    return coverage_df


def print_coverage_report(coverage_df):
    """打印覆盖率报告"""
    print("\n" + "="*80)
    print("YouTube Data Coverage Report by Season")
    print("="*80)
    print(f"{'Season':<10} {'Total':<10} {'Crawled':<10} {'Missing':<10} {'Coverage':<15}")
    print("-"*80)

    for _, row in coverage_df.iterrows():
        print(f"{int(row['season']):<10} "
              f"{int(row['total_celebrities']):<10} "
              f"{int(row['crawled_celebrities']):<10} "
              f"{int(row['missing_celebrities']):<10} "
              f"{row['coverage_rate']:.2f}%")

    print("-"*80)

    # 计算总体覆盖率
    total_celebrities = coverage_df['total_celebrities'].sum()
    total_crawled = coverage_df['crawled_celebrities'].sum()
    overall_coverage = (total_crawled / total_celebrities * 100) if total_celebrities > 0 else 0

    print(f"{'TOTAL':<10} "
          f"{int(total_celebrities):<10} "
          f"{int(total_crawled):<10} "
          f"{int(total_celebrities - total_crawled):<10} "
          f"{overall_coverage:.2f}%")
    print("="*80)

    # 统计信息
    print(f"\nStatistics:")
    print(f"  - Average coverage rate: {coverage_df['coverage_rate'].mean():.2f}%")
    print(f"  - Median coverage rate: {coverage_df['coverage_rate'].median():.2f}%")
    print(f"  - Min coverage rate: {coverage_df['coverage_rate'].min():.2f}% (Season {coverage_df.loc[coverage_df['coverage_rate'].idxmin(), 'season']:.0f})")
    print(f"  - Max coverage rate: {coverage_df['coverage_rate'].max():.2f}% (Season {coverage_df.loc[coverage_df['coverage_rate'].idxmax(), 'season']:.0f})")
    print(f"  - Seasons with 100% coverage: {len(coverage_df[coverage_df['coverage_rate'] == 100])}")
    print(f"  - Seasons with 0% coverage: {len(coverage_df[coverage_df['coverage_rate'] == 0])}")


def main():
    """主函数"""
    # 定义文件路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_path = os.path.join(project_root, 'data', 'raw', '2026_MCM_Problem_C_Data.csv')
    youtube_data_path = os.path.join(project_root, 'data', 'processed', 'youtube_data.csv')

    # 检查文件是否存在
    if not os.path.exists(raw_data_path):
        print(f"Error: Raw data file not found at {raw_data_path}")
        return

    if not os.path.exists(youtube_data_path):
        print(f"Error: YouTube data file not found at {youtube_data_path}")
        return

    # 计算覆盖率
    coverage_df = calculate_coverage_by_season(raw_data_path, youtube_data_path)

    # 打印报告
    print_coverage_report(coverage_df)

    # 保存结果到CSV
    output_path = os.path.join(project_root, 'data', 'processed', 'youtube_coverage_report.csv')
    coverage_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nCoverage report saved to: {output_path}")


if __name__ == "__main__":
    main()
