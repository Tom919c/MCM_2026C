"""
YouTube爬虫脚本 - 方案2（使用播放列表）

功能：
1. 从DWTS官方频道的赛季播放列表获取视频
2. 使用playlistItems().list()获取视频列表（1 unit）
3. 使用videos().list()批量获取统计数据（1 unit）
4. 配额消耗极低（约50 units vs 1100 units）
"""

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import time
import datetime
import os
from pathlib import Path
import httplib2
import socket
import json
import re

# ================= 配置区 =================
API_KEY = 'AIzaSyB3L5_3N2pgAlfUJfyZyJYzUs7Z3CSkBa4'
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'

# 代理配置
PROXY = 'http://127.0.0.1:7897'
TIMEOUT = 30

# 项目路径配置
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / '2026_MCM_Problem_C_Data.csv'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'processed'
OUTPUT_FILE = OUTPUT_DIR / 'youtube_data_playlist.csv'
FAILED_FILE = OUTPUT_DIR / 'youtube_failed_playlist.csv'
LOG_FILE = OUTPUT_DIR / 'youtube_log_playlist.txt'

# 赛季到播放列表ID的映射（从探索结果获取）
SEASON_PLAYLIST_MAPPING = {
    34: 'PLK1f1bOs9XycXAm_r8c_Kvm2G0OpvR422',
    33: 'PLK1f1bOs9Xyd_6IP1imJZvXSmtehLoDMD',
    32: 'PLK1f1bOs9Xyf2QAyg5LRzaZvyUfwATi-2',
    31: 'PLK1f1bOs9Xycv_vzzxiz4qy8e0gLbcppV',
    25: 'PLK1f1bOs9XyfvdtVX2Wga-G5ZfhIVy6vM',
    24: 'PLK1f1bOs9XyeW7lYtwtSpT5Mc2XPqWqOY',
    23: 'PLK1f1bOs9XyeTMyQFQqxSpHV8hs1uT1NU',
    # Season 20和19有多个播放列表，需要手动添加或使用search
    20: None,  # 需要手动查找或使用search
    3: None,   # 需要手动查找或使用search
}

# 官方频道白名单
OFFICIAL_CHANNELS = [
    'Dancing with the Stars',
    'Dancing With The Stars',
    'ABC',
    'ABC Network',
    'Good Morning America',
]

# 负面关键词
NEGATIVE_KEYWORDS = [
    'elimination', 'eliminated', 'results', 'result show',
    'controversy', 'drama', 'fight', 'argument',
    'worst', 'fail', 'mistake', 'error',
    'judges comments only', 'judges comment',
    'reaction', 'reacts',
    'interview only', 'backstage',
]

# 正面关键词
POSITIVE_KEYWORDS = [
    'performance', 'dance', 'dancing',
    'full performance', 'complete performance',
    'performs', 'dances',
]

# 爬取配置
SLEEP_INTERVAL = 0.5
RETRY_TIMES = 3


# ================= 工具函数 =================

def read_csv_with_encoding(file_path):
    """尝试多种编码读取CSV文件"""
    encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252', 'gbk']

    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"成功使用编码 {encoding} 读取文件")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"使用编码 {encoding} 读取失败: {e}")
            continue

    print("警告: 所有编码尝试失败，使用utf-8并忽略错误")
    return pd.read_csv(file_path, encoding='utf-8', errors='ignore')


def log_message(message, log_file=LOG_FILE):
    """记录日志到文件和控制台"""
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_entry + '\n')


def extract_week_from_title(title):
    """从视频标题中提取周数"""
    title_lower = title.lower()
    patterns = [
        r'week\s*#?(\d+)',
        r'wk\s*#?(\d+)',
        r'episode\s*(\d+)',
        r'ep\s*(\d+)',
        r'night\s*(\d+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, title_lower)
        if match:
            try:
                week_num = int(match.group(1))
                if 1 <= week_num <= 15:
                    return week_num
            except:
                continue
    return None


def extract_celebrity_from_title(title, celebrity_list):
    """从视频标题中匹配名人"""
    title_lower = title.lower()

    for celebrity in celebrity_list:
        # 尝试匹配全名
        if celebrity.lower() in title_lower:
            return celebrity

        # 尝试匹配姓氏
        parts = celebrity.split()
        if len(parts) >= 2:
            last_name = parts[-1]
            if len(last_name) > 3 and last_name.lower() in title_lower:
                return celebrity

    return None


def calculate_confidence_score(video_data, celebrity, partner):
    """计算视频的置信度分数"""
    score = 0
    title = video_data['title'].lower()
    channel = video_data['channel_title']

    if channel in OFFICIAL_CHANNELS:
        score += 50

    celebrity_parts = celebrity.lower().split()
    if any(part in title for part in celebrity_parts if len(part) > 2):
        score += 20

    partner_parts = partner.lower().split()
    if any(part in title for part in partner_parts if len(part) > 2):
        score += 15

    if any(keyword in title for keyword in POSITIVE_KEYWORDS):
        score += 10

    if any(keyword in title for keyword in NEGATIVE_KEYWORDS):
        score -= 30

    if 'dancing with the stars' in title or 'dwts' in title:
        score += 5

    return max(0, min(100, score))


# ================= 爬虫核心类 =================

class DWTSPlaylistCrawler:
    """DWTS播放列表爬虫"""

    def __init__(self, api_key, proxy=None, timeout=TIMEOUT):
        """初始化爬虫"""
        if proxy:
            proxy_info = httplib2.ProxyInfo(
                proxy_type=httplib2.socks.PROXY_TYPE_HTTP,
                proxy_host=proxy.split('://')[1].split(':')[0],
                proxy_port=int(proxy.split(':')[-1])
            )
            http = httplib2.Http(proxy_info=proxy_info, timeout=timeout)
            self.youtube = build(
                YOUTUBE_API_SERVICE_NAME,
                YOUTUBE_API_VERSION,
                developerKey=api_key,
                http=http
            )
            log_message(f"使用代理: {proxy}")
        else:
            http = httplib2.Http(timeout=timeout)
            self.youtube = build(
                YOUTUBE_API_SERVICE_NAME,
                YOUTUBE_API_VERSION,
                developerKey=api_key,
                http=http
            )
            log_message("不使用代理，直接连接")

        self.request_count = 0
        self.timeout = timeout

    def get_playlist_videos(self, playlist_id, max_results=50):
        """
        获取播放列表中的所有视频ID

        Args:
            playlist_id: 播放列表ID
            max_results: 每页最大结果数

        Returns:
            视频ID列表
        """
        video_ids = []

        try:
            request = self.youtube.playlistItems().list(
                part='contentDetails',
                playlistId=playlist_id,
                maxResults=max_results
            )

            while request:
                response = request.execute()
                self.request_count += 1

                for item in response['items']:
                    video_id = item['contentDetails']['videoId']
                    video_ids.append(video_id)

                # 获取下一页
                request = self.youtube.playlistItems().list_next(request, response)

            log_message(f"从播放列表获取了 {len(video_ids)} 个视频ID")
            return video_ids

        except HttpError as e:
            log_message(f"HTTP错误 {e.resp.status}: {e.content}")
            return []
        except Exception as e:
            log_message(f"获取播放列表视频失败: {e}")
            return []

    def get_videos_stats_batch(self, video_ids, batch_size=50):
        """
        批量获取视频统计数据（方案3：批量查询）

        Args:
            video_ids: 视频ID列表
            batch_size: 批量大小（最大50）

        Returns:
            视频数据列表
        """
        all_videos = []

        # 分批处理
        for i in range(0, len(video_ids), batch_size):
            batch = video_ids[i:i + batch_size]

            try:
                response = self.youtube.videos().list(
                    part='statistics,snippet',
                    id=','.join(batch)
                ).execute()

                self.request_count += 1

                for item in response['items']:
                    stats = item['statistics']
                    snippet = item['snippet']

                    data = {
                        'video_id': item['id'],
                        'title': snippet['title'],
                        'channel_title': snippet['channelTitle'],
                        'publish_date': snippet['publishedAt'],
                        'view_count': int(stats.get('viewCount', 0)),
                        'like_count': int(stats.get('likeCount', 0)),
                        'comment_count': int(stats.get('commentCount', 0)),
                    }
                    all_videos.append(data)

            except HttpError as e:
                log_message(f"HTTP错误 {e.resp.status}: {e.content}")
            except Exception as e:
                log_message(f"批量获取视频统计失败: {e}")

            time.sleep(0.1)  # 短暂延迟

        return all_videos

    def crawl_season_from_playlist(self, season, celebrity_list, partner_dict, start_week=1, end_week=10):
        """
        从播放列表爬取一个赛季的数据

        Args:
            season: 赛季号
            celebrity_list: 该赛季的名人列表
            partner_dict: {celebrity: partner} 映射
            start_week: 起始周次
            end_week: 结束周次

        Returns:
            结果列表
        """
        results = []

        # 检查是否有播放列表
        playlist_id = SEASON_PLAYLIST_MAPPING.get(season)
        if not playlist_id:
            log_message(f"Season {season} 没有播放列表映射，跳过")
            return results

        log_message(f"开始爬取 Season {season} (播放列表: {playlist_id})")

        # 1. 获取播放列表中的所有视频ID
        video_ids = self.get_playlist_videos(playlist_id)
        if not video_ids:
            log_message(f"播放列表为空或获取失败")
            return results

        # 2. 批量获取视频统计数据
        log_message(f"批量获取 {len(video_ids)} 个视频的统计数据...")
        videos = self.get_videos_stats_batch(video_ids)

        log_message(f"成功获取 {len(videos)} 个视频的数据")

        # 调试：显示前10个视频标题
        log_message(f"\n前10个视频标题示例:")
        for i, video in enumerate(videos[:10]):
            log_message(f"  {i+1}. {video['title']}")

        # 3. 解析视频并匹配到(celebrity, week)
        matched_count = 0
        for video in videos:
            # 提取周数
            week = extract_week_from_title(video['title'])
            if not week or week < start_week or week > end_week:
                continue

            # 匹配名人
            celebrity = extract_celebrity_from_title(video['title'], celebrity_list)
            if not celebrity:
                # 调试：记录未匹配的视频
                if matched_count < 5:  # 只记录前5个
                    log_message(f"  未匹配到名人: {video['title'][:80]}")
                continue

            partner = partner_dict.get(celebrity, '')

            # 计算置信度
            confidence = calculate_confidence_score(video, celebrity, partner)

            result = {
                'celebrity_name': celebrity,
                'ballroom_partner': partner,
                'season': season,
                'week': week,
                'dance_style': '',
                'yt_video_id': video['video_id'],
                'yt_video_title': video['title'],
                'yt_channel_title': video['channel_title'],
                'yt_view_count': video['view_count'],
                'yt_like_count': video['like_count'],
                'yt_comment_count': video['comment_count'],
                'yt_publish_date': video['publish_date'],
                'is_official_channel': video['channel_title'] in OFFICIAL_CHANNELS,
                'confidence_score': confidence,
                'query_used': f'playlist:{playlist_id}',
                'crawl_timestamp': datetime.datetime.now().isoformat(),
            }

            results.append(result)
            matched_count += 1
            log_message(f"  匹配: S{season} W{week} - {celebrity} (置信度: {confidence})")

        log_message(f"Season {season} 匹配成功: {matched_count} 条")
        return results

    def crawl_dataset(self, df_input, start_week=1, end_week=10, resume=True):
        """
        批量爬取数据集

        Args:
            df_input: 输入DataFrame
            start_week: 起始周次
            end_week: 结束周次
            resume: 是否断点续爬

        Returns:
            (success_df, failed_df)
        """
        results = []
        failed = []

        # 断点续爬：加载已有数据
        existing_keys = set()
        if resume and OUTPUT_FILE.exists():
            try:
                existing_df = read_csv_with_encoding(OUTPUT_FILE)
                for _, row in existing_df.iterrows():
                    key = (row['celebrity_name'], row['season'], row['week'])
                    existing_keys.add(key)
                log_message(f"断点续爬: 已加载 {len(existing_keys)} 条已爬取记录")
            except Exception as e:
                log_message(f"加载已有数据失败: {e}")

        # 按赛季分组
        seasons = df_input['season'].unique()
        log_message(f"需要爬取 {len(seasons)} 个赛季")

        for season in sorted(seasons):
            season_data = df_input[df_input['season'] == season]

            # 构建该赛季的名人列表和partner映射
            celebrity_list = season_data['celebrity_name'].tolist()
            partner_dict = dict(zip(season_data['celebrity_name'], season_data['ballroom_partner']))

            log_message(f"\n{'='*60}")
            log_message(f"处理 Season {season} ({len(celebrity_list)} 个选手)")
            log_message(f"{'='*60}")

            # 从播放列表爬取
            season_results = self.crawl_season_from_playlist(
                season, celebrity_list, partner_dict, start_week, end_week
            )

            # 保存结果
            for result in season_results:
                key = (result['celebrity_name'], result['season'], result['week'])

                if key in existing_keys:
                    continue

                results.append(result)

                # 实时保存
                try:
                    result_df = pd.DataFrame([result])
                    if OUTPUT_FILE.exists():
                        result_df.to_csv(OUTPUT_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')
                    else:
                        result_df.to_csv(OUTPUT_FILE, mode='w', header=True, index=False, encoding='utf-8-sig')
                except Exception as e:
                    log_message(f"保存失败: {e}")

            time.sleep(SLEEP_INTERVAL)

        log_message(f"\n爬取完成: 成功 {len(results)} 条, 失败 {len(failed)} 条")
        return pd.DataFrame(results), pd.DataFrame(failed)


# ================= 主流程 =================

def main():
    """主流程"""
    print("=" * 60)
    print("DWTS YouTube数据爬虫 - 方案2（播放列表）")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log_message("=" * 60)
    log_message("开始新的爬取任务（方案2：播放列表）")
    log_message("=" * 60)

    # 1. 读取原始数据
    log_message(f"读取数据文件: {RAW_DATA_PATH}")
    df_raw = read_csv_with_encoding(RAW_DATA_PATH)
    log_message(f"数据加载成功: {len(df_raw)} 行")

    # 2. 提取需要爬取的数据
    df_to_crawl = df_raw[['celebrity_name', 'ballroom_partner', 'season']].copy()

    # 只爬取有播放列表的赛季
    available_seasons = [s for s in df_to_crawl['season'].unique() if SEASON_PLAYLIST_MAPPING.get(s)]
    df_to_crawl = df_to_crawl[df_to_crawl['season'].isin(available_seasons)]

    log_message(f"准备爬取 {len(df_to_crawl)} 个选手的数据")
    log_message(f"可用赛季: {sorted(available_seasons)}")

    # 3. 初始化爬虫
    crawler = DWTSPlaylistCrawler(API_KEY, proxy=PROXY, timeout=TIMEOUT)

    # 4. 开始爬取
    df_results, df_failed = crawler.crawl_dataset(
        df_to_crawl,
        start_week=1,
        end_week=13,
        resume=True
    )

    # 5. 统计最终结果
    log_message("=" * 60)
    log_message("爬取任务完成")
    log_message(f"总API请求次数: {crawler.request_count}")

    if OUTPUT_FILE.exists():
        try:
            final_df = read_csv_with_encoding(OUTPUT_FILE)
            log_message(f"成功记录总数: {len(final_df)}")
            log_message(f"数据文件: {OUTPUT_FILE}")
        except Exception as e:
            log_message(f"读取最终数据失败: {e}")

    log_message("=" * 60)


if __name__ == "__main__":
    main()
