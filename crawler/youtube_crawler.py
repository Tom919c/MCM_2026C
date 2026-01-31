"""
YouTube爬虫脚本 - 改进版

功能：
1. 从项目CSV读取celebrity和partner信息
2. 爬取每个(celebrity, partner, season)组合的YouTube视频数据
3. 筛选标题包含名人名字和"Dancing with the Stars"的视频
4. 甄别官方频道和粉丝上传
5. 过滤负面流量
6. 支持断点续爬
7. 多编码兼容
8. 支持赛季区间配置
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

# ================= 配置区 =================
API_KEY = 'AIzaSyB77BTtiiZ3SGBJlR5B7s5HtIG1BsVr8kA'
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'

# 代理配置（如果需要）
# 格式: 'http://127.0.0.1:7890' 或 'socks5://127.0.0.1:1080'
PROXY = 'http://127.0.0.1:7897'  # 如果不需要代理，设为None；如果需要，设置为代理地址
# PROXY = 'http://127.0.0.1:7890'  # 示例：使用本地代理

# 网络超时设置（秒）
TIMEOUT = 30

# 项目路径配置
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / '2026_MCM_Problem_C_Data.csv'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'processed'
OUTPUT_FILE = OUTPUT_DIR / 'youtube_data.csv'
FAILED_FILE = OUTPUT_DIR / 'youtube_failed.csv'
LOG_FILE = OUTPUT_DIR / 'youtube_log.txt'

# 官方频道ID（用于限制搜索）
DWTS_CHANNEL_ID = 'UCLOIoa2aEGcM-z1hJx4vy4w'  # Dancing with the Stars官方频道

# 官方频道白名单
OFFICIAL_CHANNELS = [
    'Dancing with the Stars',
    'ABC',
    'ABC Network',
    'Good Morning America',
]

# 负面关键词（用于过滤非表演视频）
NEGATIVE_KEYWORDS = [
    'elimination', 'eliminated', 'results', 'result show',
    'controversy', 'drama', 'fight', 'argument',
    'worst', 'fail', 'mistake', 'error',
    'judges comments only', 'judges comment',
    'reaction', 'reacts',
    'interview only', 'backstage',
]

# 正面关键词（用于识别表演视频）
POSITIVE_KEYWORDS = [
    'performance', 'dance', 'dancing',
    'full performance', 'complete performance',
    'performs', 'dances',
]

# 爬取配置
MAX_RESULTS_PER_QUERY = 50  # 每次查询返回的最大结果数（提高到50以覆盖所有周）
SLEEP_INTERVAL = 0.5  # API请求间隔（秒）
RETRY_TIMES = 3  # 失败重试次数
MAX_SEARCH_CALLS_PER_DAY = 105  # 每天最多search调用次数（50×100=5000 units）

# 赛季区间配置
SEASON_START = 1  # 起始赛季（None表示不限制）
SEASON_END = 34    # 结束赛季（None表示不限制）
# 示例：只爬取赛季28-33的数据
# SEASON_START = 28
# SEASON_END = 33

# 缓存配置
CACHE_FILE = OUTPUT_DIR / 'youtube_cache.json'  # 缓存文件路径


# ================= 工具函数 =================

def read_csv_with_encoding(file_path):
    """
    尝试多种编码读取CSV文件

    Args:
        file_path: CSV文件路径

    Returns:
        DataFrame
    """
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

    # 最后使用utf-8并忽略错误
    print("警告: 所有编码尝试失败，使用utf-8并忽略错误")
    return pd.read_csv(file_path, encoding='utf-8', errors='ignore')


def log_message(message, log_file=LOG_FILE):
    """
    记录日志到文件和控制台

    Args:
        message: 日志消息
        log_file: 日志文件路径
    """
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}"

    # 输出到控制台
    print(log_entry)

    # 写入日志文件
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_entry + '\n')


def load_cache():
    """
    加载缓存文件

    Returns:
        缓存字典 {query: video_data}
    """
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            log_message(f"加载缓存: {len(cache)} 条记录")
            return cache
        except Exception as e:
            log_message(f"加载缓存失败: {e}")
            return {}
    return {}


def save_cache(cache):
    """
    保存缓存到文件

    Args:
        cache: 缓存字典
    """
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        log_message(f"缓存已保存: {len(cache)} 条记录")
    except Exception as e:
        log_message(f"保存缓存失败: {e}")


def construct_query(celebrity, partner, season, week, dance_style=None):
    """
    构造搜索查询关键词（简化版：只用名人名字）

    Args:
        celebrity: 名人名字
        partner: 职业舞者名字（不再使用）
        season: 赛季（不再使用）
        week: 周次（不再使用）
        dance_style: 舞蹈类型（不再使用）

    Returns:
        查询字符串
    """
    # 极简查询：只包含celebrity名字
    # 因为已限制到官方频道，官方视频标题都包含"Dancing with the Stars"
    query = f'"{celebrity}"'
    return query


def construct_fallback_query(celebrity, season, week, dance_style=None):
    """
    构造备用查询（简化版：只用名人名字）

    Args:
        celebrity: 名人名字
        season: 赛季（不再使用）
        week: 周次（不再使用）
        dance_style: 舞蹈类型（不再使用）

    Returns:
        查询字符串
    """
    # 极简查询：只包含celebrity名字
    query = f'"{celebrity}"'
    return query


def calculate_confidence_score(video_data, celebrity, partner):
    """
    计算视频的置信度分数

    Args:
        video_data: 视频数据字典
        celebrity: 名人名字
        partner: 职业舞者名字

    Returns:
        置信度分数 (0-100)
    """
    score = 0
    title = video_data['title'].lower()
    channel = video_data['channel_title']

    # 官方频道 +50分
    if channel in OFFICIAL_CHANNELS:
        score += 50

    # 标题包含celebrity名字 +20分
    celebrity_parts = celebrity.lower().split()
    if any(part in title for part in celebrity_parts if len(part) > 2):
        score += 20

    # 标题包含partner名字 +15分
    partner_parts = partner.lower().split()
    if any(part in title for part in partner_parts if len(part) > 2):
        score += 15

    # 标题包含正面关键词 +10分
    if any(keyword in title for keyword in POSITIVE_KEYWORDS):
        score += 10

    # 标题包含负面关键词 -30分
    if any(keyword in title for keyword in NEGATIVE_KEYWORDS):
        score -= 30

    # 标题包含"dancing with the stars"或"dwts" +5分
    if 'dancing with the stars' in title or 'dwts' in title:
        score += 5

    return max(0, min(100, score))  # 限制在0-100之间


def extract_week_from_title(title):
    """
    从视频标题中提取周数

    Args:
        title: 视频标题

    Returns:
        周数（int），如果无法提取返回None
    """
    import re

    title_lower = title.lower()

    # 匹配模式：Week X, Wk X, Week X:, Episode X等
    patterns = [
        r'week\s*(\d+)',
        r'wk\s*(\d+)',
        r'episode\s*(\d+)',
        r'ep\s*(\d+)',
        r'night\s*(\d+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, title_lower)
        if match:
            try:
                week_num = int(match.group(1))
                if 1 <= week_num <= 15:  # 合理的周数范围
                    return week_num
            except:
                continue

    return None


# ================= 爬虫核心类 =================

class DWTSYouTubeCrawler:
    """DWTS YouTube数据爬虫"""

    def __init__(self, api_key, proxy=None, timeout=TIMEOUT):
        """
        初始化爬虫

        Args:
            api_key: Google API密钥
            proxy: 代理地址（可选）
            timeout: 超时时间（秒）
        """
        # 配置HTTP客户端
        if proxy:
            # 如果使用代理
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
            # 不使用代理，但设置超时
            http = httplib2.Http(timeout=timeout)
            self.youtube = build(
                YOUTUBE_API_SERVICE_NAME,
                YOUTUBE_API_VERSION,
                developerKey=api_key,
                http=http
            )
            log_message("不使用代理，直接连接")

        self.request_count = 0
        self.search_count = 0  # 新增：search调用计数
        self.timeout = timeout
        self.cache = load_cache()  # 新增：加载缓存

    def search_videos(self, query, max_results=MAX_RESULTS_PER_QUERY, retry_count=0, use_cache=True, channel_id=None):
        """
        搜索视频（带缓存和配额限制）

        Args:
            query: 搜索查询
            max_results: 最大结果数
            retry_count: 当前重试次数
            use_cache: 是否使用缓存
            channel_id: 限制搜索到特定频道（可选）

        Returns:
            视频数据列表
        """
        # 检查缓存
        cache_key = f"{query}|{channel_id}" if channel_id else query
        if use_cache and cache_key in self.cache:
            log_message(f"从缓存读取: {query}")
            return self.cache[cache_key]

        # 检查配额限制
        if self.search_count >= MAX_SEARCH_CALLS_PER_DAY:
            log_message(f"已达到每日search限制 ({MAX_SEARCH_CALLS_PER_DAY})，停止搜索")
            return None

        video_data_list = []

        try:
            # 1. 搜索视频ID
            search_params = {
                'q': query,
                'part': 'id,snippet',
                'maxResults': max_results,
                'type': 'video',
                'order': 'relevance'
            }

            # 如果指定了频道ID，只搜索该频道
            if channel_id:
                search_params['channelId'] = channel_id
                log_message(f"限制搜索到频道: {channel_id}")

            search_response = self.youtube.search().list(**search_params).execute()

            self.request_count += 1
            self.search_count += 1  # 增加search计数
            log_message(f"Search调用: {self.search_count}/{MAX_SEARCH_CALLS_PER_DAY}")

            video_ids = []
            for search_result in search_response.get('items', []):
                video_ids.append(search_result['id']['videoId'])

            if not video_ids:
                return None

            # 2. 获取视频详细统计信息（方案4：字段裁剪）
            stats_response = self.youtube.videos().list(
                part='statistics,snippet',  # 去掉contentDetails，只保留需要的字段
                id=','.join(video_ids)
            ).execute()

            self.request_count += 1

            for item in stats_response['items']:
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
                video_data_list.append(data)

            # 保存到缓存
            if video_data_list:
                self.cache[cache_key] = video_data_list
                save_cache(self.cache)

        except HttpError as e:
            log_message(f"HTTP错误 {e.resp.status}: {e.content}")
            return None
        except socket.timeout as e:
            log_message(f"连接超时: {e}")
            # 重试机制
            if retry_count < RETRY_TIMES:
                log_message(f"第 {retry_count + 1} 次重试...")
                time.sleep(2 ** retry_count)  # 指数退避
                return self.search_videos(query, max_results, retry_count + 1, use_cache)
            else:
                log_message(f"重试 {RETRY_TIMES} 次后仍然失败")
                return None
        except Exception as e:
            log_message(f"搜索视频时发生错误: {e}")
            # 网络相关错误也尝试重试
            if retry_count < RETRY_TIMES and ('timeout' in str(e).lower() or 'connection' in str(e).lower()):
                log_message(f"第 {retry_count + 1} 次重试...")
                time.sleep(2 ** retry_count)  # 指数退避
                return self.search_videos(query, max_results, retry_count + 1, use_cache)
            return None

        return video_data_list

    def select_best_video(self, videos, celebrity, partner):
        """
        从多个视频中选择最佳匹配

        Args:
            videos: 视频列表
            celebrity: 名人名字
            partner: 职业舞者名字

        Returns:
            最佳视频数据（包含置信度分数）
        """
        if not videos:
            return None

        # 计算每个视频的置信度分数
        scored_videos = []
        for video in videos:
            score = calculate_confidence_score(video, celebrity, partner)
            video['confidence_score'] = score
            video['is_official_channel'] = video['channel_title'] in OFFICIAL_CHANNELS
            scored_videos.append((score, video))

        # 按置信度排序，选择最高分的
        scored_videos.sort(key=lambda x: x[0], reverse=True)
        best_video = scored_videos[0][1]

        return best_video

    def update_video_stats(self, video_id):
        """
        只更新视频统计数据（不search，只消耗1 unit）

        Args:
            video_id: 视频ID

        Returns:
            更新后的视频数据
        """
        try:
            stats_response = self.youtube.videos().list(
                part='statistics,snippet',
                id=video_id
            ).execute()

            self.request_count += 1

            if not stats_response['items']:
                return None

            item = stats_response['items'][0]
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

            return data

        except Exception as e:
            log_message(f"更新视频统计失败: {e}")
            return None

    def crawl_single_entry(self, celebrity, partner, season, week, dance_style=None):
        """
        爬取单个条目的YouTube数据

        Args:
            celebrity: 名人名字
            partner: 职业舞者名字
            season: 赛季
            week: 周次
            dance_style: 舞蹈类型（可选）

        Returns:
            包含YouTube数据的字典，如果失败返回None
        """
        # 尝试主查询（不限制频道）
        query = construct_query(celebrity, partner, season, week, dance_style)
        log_message(f"查询: {query}")

        videos = self.search_videos(query)

        # 如果主查询失败，尝试备用查询
        if not videos or len(videos) == 0:
            log_message(f"主查询无结果，尝试备用查询")
            fallback_query = construct_fallback_query(celebrity, season, week, dance_style)
            videos = self.search_videos(fallback_query)
            query = fallback_query

        if not videos or len(videos) == 0:
            log_message(f"未找到匹配视频")
            return None

        # 选择最佳视频
        best_video = self.select_best_video(videos, celebrity, partner)

        if not best_video:
            return None

        # 构造返回数据
        result = {
            'celebrity_name': celebrity,
            'ballroom_partner': partner,
            'season': season,
            'week': week,
            'dance_style': dance_style if dance_style else '',
            'yt_video_id': best_video['video_id'],
            'yt_video_title': best_video['title'],
            'yt_channel_title': best_video['channel_title'],
            'yt_view_count': best_video['view_count'],
            'yt_like_count': best_video['like_count'],
            'yt_comment_count': best_video['comment_count'],
            'yt_publish_date': best_video['publish_date'],
            'is_official_channel': best_video['is_official_channel'],
            'confidence_score': best_video['confidence_score'],
            'query_used': query,
            'crawl_timestamp': datetime.datetime.now().isoformat(),
        }

        log_message(f"找到视频: {best_video['title'][:50]}... (置信度: {best_video['confidence_score']})")

        return result

    def crawl_season_batch(self, celebrity, partner, season):
        """
        爬取名人的所有相关视频（不限制频道和赛季）

        Args:
            celebrity: 名人名字
            partner: 职业舞者名字
            season: 赛季（仅用于记录，不用于搜索）

        Returns:
            包含所有匹配视频的列表
        """
        results = []

        # 构造极简查询（只用名人名字）
        query = f'"{celebrity}"'
        log_message(f"查询: {query}")

        # 搜索该名人的所有相关视频（不限制频道和赛季）
        videos = self.search_videos(query, max_results=MAX_RESULTS_PER_QUERY)

        if not videos or len(videos) == 0:
            log_message(f"未找到匹配的视频")
            return results

        log_message(f"找到 {len(videos)} 个视频，开始筛选")

        # 筛选包含名人名字和"Dancing with the Stars"的视频
        matched_videos = []
        for video in videos:
            title_lower = video['title'].lower()
            celebrity_lower = celebrity.lower()

            # 检查标题是否包含名人名字（至少包含名字的一部分）
            celebrity_parts = celebrity_lower.split()
            has_celebrity = any(part in title_lower for part in celebrity_parts if len(part) > 2)

            # 检查标题是否包含"Dancing with the Stars"或"DWTS"
            has_dwts = 'dancing with the stars' in title_lower or 'dwts' in title_lower

            if has_celebrity and has_dwts:
                # 计算置信度
                score = calculate_confidence_score(video, celebrity, partner)
                video['confidence_score'] = score
                video['is_official_channel'] = video['channel_title'] in OFFICIAL_CHANNELS

                matched_videos.append(video)

        log_message(f"筛选后匹配 {len(matched_videos)} 个视频")

        # 收集所有匹配的视频
        for video in matched_videos:
            result = {
                'celebrity_name': celebrity,
                'ballroom_partner': partner,
                'season': season,
                'yt_video_id': video['video_id'],
                'yt_video_title': video['title'],
                'yt_channel_title': video['channel_title'],
                'yt_view_count': video['view_count'],
                'yt_like_count': video['like_count'],
                'yt_comment_count': video['comment_count'],
                'yt_publish_date': video['publish_date'],
                'is_official_channel': video['is_official_channel'],
                'confidence_score': video['confidence_score'],
                'query_used': query,
                'crawl_timestamp': datetime.datetime.now().isoformat(),
            }

            results.append(result)
            log_message(f"  -> {video['title'][:60]}... (置信度: {video['confidence_score']})")

        return results

    def crawl_dataset(self, df_input, resume=True):
        """
        批量爬取数据集（收集所有匹配视频，不限制频道）

        Args:
            df_input: 输入DataFrame（包含celebrity_name, ballroom_partner, season列）
            resume: 是否断点续爬

        Returns:
            包含成功和失败记录的元组 (success_df, failed_df)
        """
        results = []
        failed = []

        # 如果启用断点续爬，加载已有数据
        existing_combinations = set()  # 记录已完成的(celebrity, season)组合
        if resume and OUTPUT_FILE.exists():
            try:
                existing_df = read_csv_with_encoding(OUTPUT_FILE)
                for _, row in existing_df.iterrows():
                    combo_key = (row['celebrity_name'], row['season'])
                    existing_combinations.add(combo_key)
                log_message(f"断点续爬: 已完成 {len(existing_combinations)} 个(选手×赛季)组合")
            except Exception as e:
                log_message(f"加载已有数据失败: {e}")

        total_combinations = len(df_input)
        current_index = 0

        for index, row in df_input.iterrows():
            celebrity = row['celebrity_name']
            partner = row['ballroom_partner']
            season = row['season']
            current_index += 1

            # 检查配额限制
            if self.search_count >= MAX_SEARCH_CALLS_PER_DAY:
                log_message(f"已达到每日search限制，停止爬取")
                log_message(f"已完成: {current_index - 1}/{total_combinations} 个组合")
                break

            # 检查该组合是否已爬取
            combo_key = (celebrity, season)
            if combo_key in existing_combinations:
                log_message(f"[{current_index}/{total_combinations}] 跳过已完成: {celebrity} (S{season})")
                continue

            log_message(f"[{current_index}/{total_combinations}] 处理: {celebrity} & {partner} (S{season})")

            # 按赛季批量爬取（不限制频道）
            season_results = self.crawl_season_batch(celebrity, partner, season)

            # 保存所有匹配的视频
            if season_results:
                for result in season_results:
                    results.append(result)

                    # 立即保存到文件
                    try:
                        result_df = pd.DataFrame([result])
                        if OUTPUT_FILE.exists():
                            result_df.to_csv(OUTPUT_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')
                        else:
                            result_df.to_csv(OUTPUT_FILE, mode='w', header=True, index=False, encoding='utf-8-sig')
                    except Exception as e:
                        log_message(f"  -> 保存失败 - {e}")

                log_message(f"  -> 成功保存 {len(season_results)} 个视频")
            else:
                # 记录未找到视频的赛季
                failed_entry = {
                    'celebrity_name': celebrity,
                    'ballroom_partner': partner,
                    'season': season,
                    'reason': '未找到匹配视频',
                    'timestamp': datetime.datetime.now().isoformat(),
                }
                failed.append(failed_entry)

                # 立即保存失败记录
                try:
                    failed_df = pd.DataFrame([failed_entry])
                    if FAILED_FILE.exists():
                        failed_df.to_csv(FAILED_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')
                    else:
                        failed_df.to_csv(FAILED_FILE, mode='w', header=True, index=False, encoding='utf-8-sig')
                except Exception as e:
                    log_message(f"  -> 保存失败记录时出错: {e}")

            # API请求间隔
            time.sleep(SLEEP_INTERVAL)

        log_message(f"爬取完成: 成功 {len(results)} 条, 失败 {len(failed)} 条")

        return pd.DataFrame(results), pd.DataFrame(failed)


# ================= 主流程 =================

def main():
    """主流程"""
    print("=" * 60)
    print("DWTS YouTube数据爬虫 - 开始运行")
    print("=" * 60)

    # 确保输出目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 初始化日志
    log_message("=" * 60)
    log_message("开始新的爬取任务")
    log_message("=" * 60)

    # 1. 读取原始数据
    log_message(f"读取数据文件: {RAW_DATA_PATH}")
    df_raw = read_csv_with_encoding(RAW_DATA_PATH)
    log_message(f"数据加载成功: {len(df_raw)} 行")

    # 2. 提取需要爬取的数据
    # 只保留需要的列
    df_to_crawl = df_raw[['celebrity_name', 'ballroom_partner', 'season']].copy()

    # 根据赛季区间配置筛选数据（只用于限制爬取范围）
    if SEASON_START is not None or SEASON_END is not None:
        if SEASON_START is not None and SEASON_END is not None:
            df_to_crawl = df_to_crawl[
                (df_to_crawl['season'] >= SEASON_START) &
                (df_to_crawl['season'] <= SEASON_END)
            ]
            log_message(f"筛选赛季区间: {SEASON_START} - {SEASON_END}")
        elif SEASON_START is not None:
            df_to_crawl = df_to_crawl[df_to_crawl['season'] >= SEASON_START]
            log_message(f"筛选赛季: >= {SEASON_START}")
        elif SEASON_END is not None:
            df_to_crawl = df_to_crawl[df_to_crawl['season'] <= SEASON_END]
            log_message(f"筛选赛季: <= {SEASON_END}")

    # 可选: 只爬取前N行（用于测试）
    # df_to_crawl = df_to_crawl.head(5)

    log_message(f"准备爬取 {len(df_to_crawl)} 个(选手×赛季)组合的数据")

    # 3. 初始化爬虫
    if API_KEY == 'YOUR_GOOGLE_API_KEY_HERE':
        log_message("错误: 请先配置Google API Key")
        return

    crawler = DWTSYouTubeCrawler(API_KEY, proxy=PROXY, timeout=TIMEOUT)

    # 4. 开始爬取
    # 参数说明:
    # - resume: 是否断点续爬（默认True）
    df_results, df_failed = crawler.crawl_dataset(
        df_to_crawl,
        resume=True
    )

    # 5. 统计最终结果
    # 注意：数据已经在爬取过程中实时保存了
    log_message("=" * 60)
    log_message("爬取任务完成")
    log_message(f"总API请求次数: {crawler.request_count}")
    log_message(f"Search调用次数: {crawler.search_count}/{MAX_SEARCH_CALLS_PER_DAY}")
    log_message(f"缓存命中数: {len(crawler.cache)} 条")

    # 统计最终保存的数据
    if OUTPUT_FILE.exists():
        try:
            final_df = read_csv_with_encoding(OUTPUT_FILE)
            log_message(f"成功记录总数: {len(final_df)}")
            log_message(f"数据文件: {OUTPUT_FILE}")
        except Exception as e:
            log_message(f"读取最终数据失败: {e}")

    if FAILED_FILE.exists():
        try:
            final_failed = read_csv_with_encoding(FAILED_FILE)
            log_message(f"失败记录总数: {len(final_failed)}")
            log_message(f"失败文件: {FAILED_FILE}")
        except Exception as e:
            log_message(f"读取失败记录失败: {e}")

    log_message("=" * 60)


if __name__ == "__main__":
    main()
