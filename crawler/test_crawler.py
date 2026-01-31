"""
YouTube爬虫测试脚本

测试爬取指定名人的数据
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from crawler.youtube_crawler import (
    DWTSYouTubeCrawler,
    API_KEY,
    PROXY,
    TIMEOUT,
    OUTPUT_DIR,
    log_message
)

# 测试配置
TEST_OUTPUT_FILE = OUTPUT_DIR / 'youtube_data_test.csv'
TEST_LOG_FILE = OUTPUT_DIR / 'youtube_log_test.txt'

# 测试名人列表
TEST_CELEBRITIES = [
    'Karamo Brown',
    'Hannah Brown',
    'Anne Heche',
    'Jesse Metcalfe',
    'Chrishell Stause'
]


def main():
    """测试主流程"""
    print("=" * 60)
    print("YouTube爬虫测试")
    print("=" * 60)

    # 确保输出目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log_message("=" * 60, log_file=TEST_LOG_FILE)
    log_message("开始测试爬取", log_file=TEST_LOG_FILE)
    log_message(f"测试名人: {', '.join(TEST_CELEBRITIES)}", log_file=TEST_LOG_FILE)
    log_message("=" * 60, log_file=TEST_LOG_FILE)

    # 从原始数据中筛选测试名人
    raw_data_path = project_root / 'data' / 'raw' / '2026_MCM_Problem_C_Data.csv'

    try:
        # 读取原始数据
        df_raw = pd.read_csv(raw_data_path, encoding='utf-8-sig')
        log_message(f"原始数据加载成功: {len(df_raw)} 行", log_file=TEST_LOG_FILE)

        # 筛选测试名人
        df_test = df_raw[df_raw['celebrity_name'].isin(TEST_CELEBRITIES)].copy()

        if len(df_test) == 0:
            log_message("错误: 未找到测试名人的数据", log_file=TEST_LOG_FILE)
            print("错误: 未找到测试名人的数据")
            return

        log_message(f"找到 {len(df_test)} 个测试选手:", log_file=TEST_LOG_FILE)
        for _, row in df_test.iterrows():
            log_message(f"  - {row['celebrity_name']} (Season {row['season']}) & {row['ballroom_partner']}",
                       log_file=TEST_LOG_FILE)

        # 初始化爬虫
        log_message("\n初始化爬虫...", log_file=TEST_LOG_FILE)
        crawler = DWTSYouTubeCrawler(API_KEY, proxy=PROXY, timeout=TIMEOUT)

        # 开始爬取
        log_message("\n开始爬取数据...", log_file=TEST_LOG_FILE)

        results = []
        failed = []

        for index, row in df_test.iterrows():
            celebrity = row['celebrity_name']
            partner = row['ballroom_partner']
            season = row['season']

            log_message(f"\n{'='*60}", log_file=TEST_LOG_FILE)
            log_message(f"爬取: {celebrity} (Season {season}) & {partner}", log_file=TEST_LOG_FILE)
            log_message(f"{'='*60}", log_file=TEST_LOG_FILE)

            # 使用crawl_season_batch方法
            season_results = crawler.crawl_season_batch(
                celebrity, partner, season
            )

            if season_results:
                results.extend(season_results)
                log_message(f"成功爬取 {len(season_results)} 条记录", log_file=TEST_LOG_FILE)

                # 显示前3条结果
                for i, result in enumerate(season_results[:3]):
                    log_message(f"  [{i+1}] {result['yt_video_title'][:60]}...",
                               log_file=TEST_LOG_FILE)
                    log_message(f"      频道: {result['yt_channel_title']}", log_file=TEST_LOG_FILE)
                    log_message(f"      播放量: {result['yt_view_count']:,}", log_file=TEST_LOG_FILE)
                    log_message(f"      官方频道: {result['is_official_channel']}", log_file=TEST_LOG_FILE)
                    log_message(f"      置信度: {result['confidence_score']}", log_file=TEST_LOG_FILE)
            else:
                log_message(f"未找到数据", log_file=TEST_LOG_FILE)
                failed.append({
                    'celebrity_name': celebrity,
                    'season': season,
                    'reason': '未找到匹配视频'
                })

        # 保存结果
        if results:
            df_results = pd.DataFrame(results)
            df_results.to_csv(TEST_OUTPUT_FILE, index=False, encoding='utf-8-sig')

            log_message(f"\n{'='*60}", log_file=TEST_LOG_FILE)
            log_message(f"测试完成", log_file=TEST_LOG_FILE)
            log_message(f"成功: {len(results)} 条", log_file=TEST_LOG_FILE)
            log_message(f"失败: {len(failed)} 条", log_file=TEST_LOG_FILE)
            log_message(f"API请求次数: {crawler.request_count}", log_file=TEST_LOG_FILE)
            log_message(f"Search调用次数: {crawler.search_count}", log_file=TEST_LOG_FILE)
            log_message(f"结果已保存到: {TEST_OUTPUT_FILE}", log_file=TEST_LOG_FILE)
            log_message(f"{'='*60}", log_file=TEST_LOG_FILE)

            # 统计信息
            print(f"\n{'='*60}")
            print(f"测试完成!")
            print(f"{'='*60}")
            print(f"成功: {len(results)} 条")
            print(f"失败: {len(failed)} 条")
            print(f"API请求次数: {crawler.request_count}")
            print(f"Search调用次数: {crawler.search_count}")
            print(f"\n结果文件: {TEST_OUTPUT_FILE}")
            print(f"日志文件: {TEST_LOG_FILE}")

            # 显示官方频道统计
            official_count = sum(1 for r in results if r['is_official_channel'])
            print(f"\n官方频道视频: {official_count}/{len(results)} ({official_count/len(results)*100:.1f}%)")

            # 显示频道分布
            channels = {}
            for r in results:
                ch = r['yt_channel_title']
                channels[ch] = channels.get(ch, 0) + 1

            print(f"\n频道分布:")
            for ch, count in sorted(channels.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {ch}: {count} 个视频")

        else:
            log_message(f"\n错误: 未爬取到任何数据", log_file=TEST_LOG_FILE)
            print("\n错误: 未爬取到任何数据")

    except Exception as e:
        log_message(f"\n错误: {e}", log_file=TEST_LOG_FILE)
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
