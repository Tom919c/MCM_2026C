# YouTube爬虫使用说明

## 功能特性

1. **自动读取项目数据**：从`data/raw/2026_MCM_Problem_C_Data.csv`读取celebrity和partner信息
2. **官方频道识别**：优先选择ABC、Dancing with the Stars等官方频道的视频
3. **负面流量过滤**：自动过滤淘汰视频、争议视频等负面内容
4. **置信度评分**：为每个视频计算置信度分数（0-100）
5. **断点续爬**：支持中断后继续爬取，避免重复请求
6. **多编码兼容**：自动尝试多种编码读取CSV文件
7. **详细日志**：记录所有操作到日志文件
8. **代理支持**：支持HTTP/SOCKS5代理（适用于网络受限环境）
9. **自动重试**：网络错误时自动重试，使用指数退避策略
10. **智能缓存**：自动缓存search结果，避免重复搜索（节省90%配额）
11. **配额限制**：每日search次数限制，防止配额耗尽
12. **实时保存**：每爬取一条立即保存，程序中断不丢失数据

## 配额优化（重要）

### 配额消耗对比

**优化前**：
- 每个(选手, 周)都search：300次 × 101 units = **30,300 units**
- 超过每日配额10,000 units

**优化后**：
- ✅ **缓存机制**：search结果永久缓存，第二次运行0消耗
- ✅ **配额限制**：最多80次search = **8,000 units**（安全范围）
- ✅ **实时保存**：中断后已爬取数据不丢失

### 配额使用策略

1. **第一天**：爬取80条（达到限制自动停止）
2. **第二天**：继续爬取80条（断点续爬）
3. **第三天**：继续...
4. **后续运行**：从缓存读取，只用videos().list()更新统计（1 unit/次）

## 网络配置（重要）

### 如果遇到连接超时错误

如果看到类似以下错误：
```
[WinError 10060] 由于连接方在一段时间后没有正确答复或连接的主机没有反应，连接尝试失败。
```

这表示无法直接访问YouTube API。解决方法：

**方法1：配置代理**

编辑`youtube_crawler.py`，修改第17-18行：

```python
# 不使用代理（默认）
PROXY = None

# 使用HTTP代理
PROXY = 'http://127.0.0.1:7890'

# 使用SOCKS5代理
PROXY = 'socks5://127.0.0.1:1080'
```

**方法2：调整超时时间**

如果网络较慢，可以增加超时时间（第21行）：

```python
TIMEOUT = 60  # 从30秒增加到60秒
```

**方法3：使用VPN**

确保VPN已连接并正常工作。

## 输出文件

- `data/processed/youtube_data.csv`：成功爬取的数据
- `data/processed/youtube_failed.csv`：失败的查询记录
- `data/processed/youtube_log.txt`：详细日志
- `data/processed/youtube_cache.json`：search结果缓存（自动管理）

## 数据字段说明

### 成功数据字段
- `celebrity_name`：名人名字
- `ballroom_partner`：职业舞者名字
- `season`：赛季
- `week`：周次
- `dance_style`：舞蹈类型（暂未使用）
- `yt_video_id`：YouTube视频ID
- `yt_video_title`：视频标题
- `yt_channel_title`：频道名称
- `yt_view_count`：播放量
- `yt_like_count`：点赞数
- `yt_comment_count`：评论数
- `yt_publish_date`：发布日期
- `is_official_channel`：是否官方频道（True/False）
- `confidence_score`：置信度分数（0-100）
- `query_used`：使用的查询关键词
- `crawl_timestamp`：爬取时间戳

## 使用方法

### 1. 配置API Key

编辑`youtube_crawler.py`，修改第22行的API_KEY：

```python
API_KEY = 'YOUR_GOOGLE_API_KEY_HERE'
```

### 2. 运行爬虫

```bash
# 激活conda环境
activate MCM

# 运行爬虫
python crawler/youtube_crawler.py
```

### 3. 测试模式

如果想先测试，可以修改main()函数中的以下行：

```python
# 只爬取特定赛季
df_to_crawl = df_to_crawl[df_to_crawl['season'].isin([1, 2, 3])]

# 只爬取前5个选手
df_to_crawl = df_to_crawl.head(5)

# 只爬取前3周
df_results, df_failed = crawler.crawl_dataset(
    df_to_crawl,
    start_week=1,
    end_week=3,  # 改为3
    resume=True
)
```

## 配置参数

在脚本顶部可以调整以下参数：

```python
# API配置
API_KEY = 'YOUR_API_KEY'       # Google API密钥

# 代理配置
PROXY = None                    # 不使用代理
# PROXY = 'http://127.0.0.1:7890'  # HTTP代理示例
# PROXY = 'socks5://127.0.0.1:1080'  # SOCKS5代理示例

# 网络配置
TIMEOUT = 30                    # 超时时间（秒）

# 爬取配置
MAX_RESULTS_PER_QUERY = 5      # 每次查询返回的最大结果数
SLEEP_INTERVAL = 0.5           # API请求间隔（秒）
RETRY_TIMES = 3                # 失败重试次数
```

## 官方频道白名单

当前识别的官方频道：
- Dancing with the Stars
- ABC
- ABC Network
- Good Morning America

可以在`OFFICIAL_CHANNELS`列表中添加更多官方频道。

## 负面关键词过滤

当前过滤的负面关键词：
- elimination, eliminated, results
- controversy, drama, fight
- worst, fail, mistake
- judges comments only
- reaction, interview only

可以在`NEGATIVE_KEYWORDS`列表中添加更多关键词。

## 置信度评分规则

- 官方频道：+50分
- 标题包含celebrity名字：+20分
- 标题包含partner名字：+15分
- 标题包含正面关键词（performance等）：+10分
- 标题包含"DWTS"或"Dancing with the Stars"：+5分
- 标题包含负面关键词：-30分

最终分数限制在0-100之间。

## 注意事项

1. **API配额限制**：YouTube API有每日配额限制，建议分批爬取
2. **请求间隔**：默认每次请求间隔0.5秒，避免触发限流
3. **断点续爬**：如果中断，再次运行会自动跳过已爬取的数据
4. **数据验证**：建议检查`confidence_score`较低的记录，可能需要手动验证

## 故障排除

### 问题1：连接超时（WinError 10060）
**症状**：`由于连接方在一段时间后没有正确答复或连接的主机没有反应，连接尝试失败`

**解决方案**：
1. 配置代理（见上方"网络配置"部分）
2. 增加超时时间：`TIMEOUT = 60`
3. 确保VPN已连接
4. 检查防火墙设置

### 问题2：编码错误
**解决方案**：脚本会自动尝试多种编码，如果仍然失败，检查CSV文件编码。

### 问题3：API配额超限
**症状**：HTTP错误403或429

**解决方案**：
1. 等待24小时后重试
2. 申请更高配额
3. 使用多个API Key轮换

### 问题4：找不到匹配视频
**解决方案**：检查`youtube_failed.csv`，可能需要手动搜索或调整查询策略。

### 问题5：置信度分数低
**解决方案**：检查视频标题和频道，可能是粉丝上传或非表演视频。建议人工验证置信度<50的记录。
