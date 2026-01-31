"""
DWTS官方频道探索脚本

功能：
1. 探索3个官方频道的播放列表结构
2. 查找是否有按赛季组织的播放列表
3. 记录可用的播放列表信息
"""

import sys
from pathlib import Path
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from googleapiclient.discovery import build
import httplib2

# ================= 配置 =================
API_KEY = 'AIzaSyB3L5_3N2pgAlfUJfyZyJYzUs7Z3CSkBa4'  # 使用旧的API Key
PROXY = 'http://127.0.0.1:7897'
TIMEOUT = 30

# 官方频道信息
OFFICIAL_CHANNELS = {
    'Dancing with the Stars': {
        'url': 'https://www.youtube.com/dancingwiththestars',
        'channel_id': 'UCLOIoa2aEGcM-z1hJx4vy4w',  # 官方频道ID
        'username': 'dancingwiththestars'
    },
    'ABC': {
        'url': 'https://www.youtube.com/ABC',
        'channel_id': 'UCBi2mrWuNuyYy4gbM6fU18Q',  # ABC官方频道ID
        'username': 'ABC'
    },
    'Good Morning America': {
        'url': 'https://www.youtube.com/channel/UCH1oRy1dINbMVp3UFWrKP0w',
        'channel_id': 'UCH1oRy1dINbMVp3UFWrKP0w',
        'username': None
    }
}

OUTPUT_FILE = project_root / 'crawler' / 'channel_playlists.json'


# ================= 初始化YouTube客户端 =================

def init_youtube_client():
    """初始化YouTube API客户端"""
    if PROXY:
        proxy_info = httplib2.ProxyInfo(
            proxy_type=httplib2.socks.PROXY_TYPE_HTTP,
            proxy_host=PROXY.split('://')[1].split(':')[0],
            proxy_port=int(PROXY.split(':')[-1])
        )
        http = httplib2.Http(proxy_info=proxy_info, timeout=TIMEOUT)
        youtube = build('youtube', 'v3', developerKey=API_KEY, http=http)
        print(f"使用代理: {PROXY}")
    else:
        http = httplib2.Http(timeout=TIMEOUT)
        youtube = build('youtube', 'v3', developerKey=API_KEY, http=http)
        print("不使用代理")

    return youtube


# ================= 频道探索函数 =================

def get_channel_id_by_username(youtube, username):
    """通过用户名获取频道ID"""
    try:
        response = youtube.channels().list(
            part='id,snippet',
            forUsername=username
        ).execute()

        if response['items']:
            return response['items'][0]['id']
        return None
    except Exception as e:
        print(f"获取频道ID失败: {e}")
        return None


def get_channel_info(youtube, channel_id):
    """获取频道信息"""
    try:
        response = youtube.channels().list(
            part='snippet,contentDetails,statistics',
            id=channel_id
        ).execute()

        if response['items']:
            item = response['items'][0]
            return {
                'channel_id': item['id'],
                'title': item['snippet']['title'],
                'description': item['snippet']['description'][:200],
                'subscriber_count': item['statistics'].get('subscriberCount', 'N/A'),
                'video_count': item['statistics'].get('videoCount', 'N/A'),
            }
        return None
    except Exception as e:
        print(f"获取频道信息失败: {e}")
        return None


def get_channel_playlists(youtube, channel_id, max_results=50):
    """获取频道的所有播放列表"""
    playlists = []

    try:
        request = youtube.playlists().list(
            part='snippet,contentDetails',
            channelId=channel_id,
            maxResults=max_results
        )

        while request:
            response = request.execute()

            for item in response['items']:
                playlist_info = {
                    'playlist_id': item['id'],
                    'title': item['snippet']['title'],
                    'description': item['snippet'].get('description', '')[:200],
                    'item_count': item['contentDetails']['itemCount'],
                    'published_at': item['snippet']['publishedAt'],
                }
                playlists.append(playlist_info)

            # 获取下一页
            request = youtube.playlists().list_next(request, response)

        return playlists
    except Exception as e:
        print(f"获取播放列表失败: {e}")
        return []


def analyze_playlists_for_seasons(playlists):
    """分析播放列表，查找按赛季组织的列表"""
    season_playlists = []

    for playlist in playlists:
        title = playlist['title'].lower()

        # 查找包含"season"关键词的播放列表
        if 'season' in title or 'series' in title:
            season_playlists.append(playlist)

    return season_playlists


# ================= 主流程 =================

def main():
    """主流程"""
    print("=" * 60)
    print("DWTS官方频道探索")
    print("=" * 60)

    youtube = init_youtube_client()

    all_results = {}
    api_calls = 0

    for channel_name, channel_info in OFFICIAL_CHANNELS.items():
        print(f"\n{'=' * 60}")
        print(f"探索频道: {channel_name}")
        print(f"{'=' * 60}")

        # 获取频道ID
        channel_id = channel_info['channel_id']
        if not channel_id and channel_info['username']:
            print(f"通过用户名获取频道ID: {channel_info['username']}")
            channel_id = get_channel_id_by_username(youtube, channel_info['username'])
            api_calls += 1

        if not channel_id:
            print(f"无法获取频道ID，跳过")
            continue

        print(f"频道ID: {channel_id}")

        # 获取频道信息
        print(f"\n获取频道信息...")
        info = get_channel_info(youtube, channel_id)
        api_calls += 1

        if info:
            print(f"  标题: {info['title']}")
            print(f"  订阅数: {info['subscriber_count']}")
            print(f"  视频数: {info['video_count']}")

        # 获取播放列表
        print(f"\n获取播放列表...")
        playlists = get_channel_playlists(youtube, channel_id)
        api_calls += 1

        print(f"  找到 {len(playlists)} 个播放列表")

        # 分析播放列表
        season_playlists = analyze_playlists_for_seasons(playlists)

        if season_playlists:
            print(f"\n  找到 {len(season_playlists)} 个可能按赛季组织的播放列表:")
            for pl in season_playlists:
                print(f"    - {pl['title']} (ID: {pl['playlist_id']}, {pl['item_count']} 个视频)")
        else:
            print(f"\n  未找到明确按赛季组织的播放列表")

        # 显示所有播放列表（前10个）
        print(f"\n  所有播放列表（前10个）:")
        for pl in playlists[:10]:
            print(f"    - {pl['title']} ({pl['item_count']} 个视频)")

        if len(playlists) > 10:
            print(f"    ... 还有 {len(playlists) - 10} 个播放列表")

        # 保存结果
        all_results[channel_name] = {
            'channel_id': channel_id,
            'channel_info': info,
            'total_playlists': len(playlists),
            'season_playlists': season_playlists,
            'all_playlists': playlists
        }

    # 保存到JSON文件
    print(f"\n{'=' * 60}")
    print(f"保存结果到: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"探索完成")
    print(f"总API调用次数: {api_calls}")
    print(f"预估配额消耗: {api_calls} units")
    print(f"{'=' * 60}")

    # 总结
    print(f"\n总结:")
    total_season_playlists = sum(len(r['season_playlists']) for r in all_results.values())
    if total_season_playlists > 0:
        print(f"✅ 找到 {total_season_playlists} 个可能按赛季组织的播放列表")
        print(f"✅ 可以考虑实施方案2（使用播放列表）")
    else:
        print(f"❌ 未找到明确按赛季组织的播放列表")
        print(f"❌ 建议继续使用优化后的方案1（按赛季search）")


if __name__ == "__main__":
    main()
