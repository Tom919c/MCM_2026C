"""
Facebook Fans Features Module

从Facebook粉丝数据中提取特征，添加到训练数据中
"""

import pandas as pd
import numpy as np
import os


def load_facebook_fans_data(facebook_data_path):
    """
    读取Facebook粉丝数据

    Args:
        facebook_data_path: Facebook数据文件路径

    Returns:
        celeb_fans_dict: 名人粉丝数字典 {celebrity_name: fans_count}
        pro_fans_dict: 舞者粉丝数字典 {ballroom_partner: fans_count}
    """
    print("加载Facebook粉丝数据...")

    # 使用多编码尝试机制读取CSV
    encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252', 'gbk']
    fb_df = None
    for encoding in encodings:
        try:
            # 手动指定列名，因为原文件有重复的列名
            fb_df = pd.read_csv(
                facebook_data_path,
                encoding=encoding,
                names=['celebrity_name', 'celeb_fans', 'ballroom_partner', 'pro_fans'],
                skiprows=1  # 跳过原始表头
            )
            break
        except UnicodeDecodeError:
            continue

    if fb_df is None:
        fb_df = pd.read_csv(
            facebook_data_path,
            encoding='utf-8',
            errors='ignore',
            names=['celebrity_name', 'celeb_fans', 'ballroom_partner', 'pro_fans'],
            skiprows=1
        )

    print(f"  - 加载了 {len(fb_df)} 条Facebook粉丝记录")

    # 构建名人粉丝数字典
    celeb_fans_dict = {}
    for _, row in fb_df.iterrows():
        celebrity_name = row['celebrity_name']
        fans_count = row['celeb_fans']

        # 如果同一个名人出现多次，取最大值
        if celebrity_name in celeb_fans_dict:
            celeb_fans_dict[celebrity_name] = max(celeb_fans_dict[celebrity_name], fans_count)
        else:
            celeb_fans_dict[celebrity_name] = fans_count

    # 构建舞者粉丝数字典
    pro_fans_dict = {}
    for _, row in fb_df.iterrows():
        pro_name = row['ballroom_partner']
        fans_count = row['pro_fans']

        # 如果同一个舞者出现多次，取最大值
        if pro_name in pro_fans_dict:
            pro_fans_dict[pro_name] = max(pro_fans_dict[pro_name], fans_count)
        else:
            pro_fans_dict[pro_name] = fans_count

    print(f"  - 唯一名人数: {len(celeb_fans_dict)}")
    print(f"  - 唯一舞者数: {len(pro_fans_dict)}")

    return celeb_fans_dict, pro_fans_dict


def add_facebook_fans_features(config, datas):
    """
    将Facebook粉丝数特征添加到train_data中

    在X_celeb中添加1个新特征：
    - celeb_facebook_fans_norm: Z-score归一化的名人粉丝数

    在X_pro中添加1个新特征：
    - pro_facebook_fans_norm: Z-score归一化的舞者粉丝数

    Args:
        config: 配置字典
        datas: 数据字典（包含train_data和index_maps）

    Returns:
        datas: 更新后的数据字典
    """
    print("\n" + "=" * 60)
    print("添加Facebook粉丝数特征到train_data")
    print("=" * 60)

    # 获取Facebook数据路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    facebook_data_path = os.path.join(project_root, 'data', 'raw', 'facebook_fans.csv')

    if not os.path.exists(facebook_data_path):
        print(f"警告: Facebook数据文件不存在: {facebook_data_path}")
        print("跳过Facebook粉丝数特征添加")
        return datas

    # Step 1: 加载Facebook粉丝数据
    celeb_fans_dict, pro_fans_dict = load_facebook_fans_data(facebook_data_path)

    # Step 2: 获取train_data和index_maps
    train_data = datas['train_data']
    index_maps = datas['index_maps']

    n_celebs = train_data['n_celebs']
    n_pros = train_data['n_pros']

    # Step 3: 为每个名人匹配粉丝数
    print("\n匹配名人粉丝数...")
    celeb_fans = np.zeros(n_celebs, dtype=np.float32)
    idx_to_celeb = index_maps['idx_to_celeb']

    matched_celebs = 0
    for celeb_idx in range(n_celebs):
        celeb_name = idx_to_celeb[celeb_idx]
        if celeb_name in celeb_fans_dict:
            celeb_fans[celeb_idx] = celeb_fans_dict[celeb_name]
            matched_celebs += 1

    print(f"  - 成功匹配 {matched_celebs} / {n_celebs} 名人 ({matched_celebs / n_celebs * 100:.2f}%)")

    # Step 4: 为每个舞者匹配粉丝数
    print("\n匹配舞者粉丝数...")
    pro_fans = np.zeros(n_pros, dtype=np.float32)
    idx_to_pro = index_maps['idx_to_pro']

    matched_pros = 0
    for pro_idx in range(n_pros):
        pro_name = idx_to_pro[pro_idx]
        if pro_name in pro_fans_dict:
            pro_fans[pro_idx] = pro_fans_dict[pro_name]
            matched_pros += 1

    print(f"  - 成功匹配 {matched_pros} / {n_pros} 舞者 ({matched_pros / n_pros * 100:.2f}%)")

    # Step 5: Z-score归一化
    print("\n使用Z-score归一化粉丝数...")

    # 名人粉丝数归一化
    celeb_fans_mean = celeb_fans.mean()
    celeb_fans_std = celeb_fans.std()

    if celeb_fans_std > 1e-8:
        celeb_fans_norm = (celeb_fans - celeb_fans_mean) / celeb_fans_std
    else:
        celeb_fans_norm = celeb_fans - celeb_fans_mean

    print(f"  - 名人粉丝数: mean={celeb_fans_mean:.2f}, std={celeb_fans_std:.2f}")

    # 舞者粉丝数归一化
    pro_fans_mean = pro_fans.mean()
    pro_fans_std = pro_fans.std()

    if pro_fans_std > 1e-8:
        pro_fans_norm = (pro_fans - pro_fans_mean) / pro_fans_std
    else:
        pro_fans_norm = pro_fans - pro_fans_mean

    print(f"  - 舞者粉丝数: mean={pro_fans_mean:.2f}, std={pro_fans_std:.2f}")

    # Step 6: 添加到X_celeb
    print("\n添加特征到X_celeb...")
    train_data['X_celeb'] = np.column_stack([
        train_data['X_celeb'],
        celeb_fans_norm.reshape(-1, 1)
    ])
    train_data['X_celeb_names'].append('celeb_facebook_fans_norm')

    print(f"  - X_celeb shape: {train_data['X_celeb'].shape}")
    print(f"  - X_celeb_names: {len(train_data['X_celeb_names'])} 个特征")

    # Step 7: 添加到X_pro
    print("\n添加特征到X_pro...")
    train_data['X_pro'] = np.column_stack([
        train_data['X_pro'],
        pro_fans_norm.reshape(-1, 1)
    ])
    train_data['X_pro_names'].append('pro_facebook_fans_norm')

    print(f"  - X_pro shape: {train_data['X_pro'].shape}")
    print(f"  - X_pro_names: {len(train_data['X_pro_names'])} 个特征")

    # 保存归一化参数
    if 'facebook_normalization_params' not in datas:
        datas['facebook_normalization_params'] = {}

    datas['facebook_normalization_params']['celeb_fans'] = {
        'mean': float(celeb_fans_mean),
        'std': float(celeb_fans_std)
    }
    datas['facebook_normalization_params']['pro_fans'] = {
        'mean': float(pro_fans_mean),
        'std': float(pro_fans_std)
    }

    print("\nFacebook粉丝数特征添加完成！")
    print("=" * 60)

    return datas
