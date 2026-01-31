"""
Test Facebook Fans Features Addition

测试Facebook粉丝数特征添加功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
from src.facebook_features import add_facebook_fans_features


def load_datas(data_path):
    """加载已保存的datas字典"""
    print(f"加载datas字典: {data_path}")
    with open(data_path, 'rb') as f:
        datas = pickle.load(f)
    return datas


def print_facebook_feature_stats(datas):
    """打印Facebook粉丝数特征统计信息"""
    train_data = datas['train_data']
    X_celeb = train_data['X_celeb']
    X_celeb_names = train_data['X_celeb_names']
    X_pro = train_data['X_pro']
    X_pro_names = train_data['X_pro_names']

    print("\n" + "=" * 60)
    print("Facebook粉丝数特征统计")
    print("=" * 60)

    # 统计名人粉丝数特征
    if 'celeb_facebook_fans_norm' in X_celeb_names:
        celeb_fans_idx = X_celeb_names.index('celeb_facebook_fans_norm')
        celeb_fans = X_celeb[:, celeb_fans_idx]

        print(f"\n1. celeb_facebook_fans_norm:")
        print(f"   - 均值: {celeb_fans.mean():.4f}")
        print(f"   - 标准差: {celeb_fans.std():.4f}")
        print(f"   - 最小值: {celeb_fans.min():.4f}")
        print(f"   - 最大值: {celeb_fans.max():.4f}")
        print(f"   - 中位数: {np.median(celeb_fans):.4f}")
        print(f"   - 非零值数量: {np.sum(celeb_fans != 0)} / {len(celeb_fans)}")
    else:
        print("\n未找到celeb_facebook_fans_norm特征")

    # 统计舞者粉丝数特征
    if 'pro_facebook_fans_norm' in X_pro_names:
        pro_fans_idx = X_pro_names.index('pro_facebook_fans_norm')
        pro_fans = X_pro[:, pro_fans_idx]

        print(f"\n2. pro_facebook_fans_norm:")
        print(f"   - 均值: {pro_fans.mean():.4f}")
        print(f"   - 标准差: {pro_fans.std():.4f}")
        print(f"   - 最小值: {pro_fans.min():.4f}")
        print(f"   - 最大值: {pro_fans.max():.4f}")
        print(f"   - 中位数: {np.median(pro_fans):.4f}")
        print(f"   - 非零值数量: {np.sum(pro_fans != 0)} / {len(pro_fans)}")
    else:
        print("\n未找到pro_facebook_fans_norm特征")

    # 打印归一化参数
    if 'facebook_normalization_params' in datas:
        print("\n" + "=" * 60)
        print("归一化参数")
        print("=" * 60)
        params = datas['facebook_normalization_params']

        if 'celeb_fans' in params:
            print(f"\n名人粉丝数:")
            print(f"  - 原始均值: {params['celeb_fans']['mean']:.2f}")
            print(f"  - 原始标准差: {params['celeb_fans']['std']:.2f}")

        if 'pro_fans' in params:
            print(f"\n舞者粉丝数:")
            print(f"  - 原始均值: {params['pro_fans']['mean']:.2f}")
            print(f"  - 原始标准差: {params['pro_fans']['std']:.2f}")

    print("=" * 60)


def main():
    """主函数"""
    # 定义文件路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, 'data', 'processed', 'datas.pkl')

    # 检查文件是否存在
    if not os.path.exists(data_path):
        print(f"错误: datas文件不存在: {data_path}")
        print("请先运行训练数据构建流程")
        return

    # 加载datas字典
    datas = load_datas(data_path)

    print(f"\n原始X_celeb shape: {datas['train_data']['X_celeb'].shape}")
    print(f"原始X_celeb_names: {len(datas['train_data']['X_celeb_names'])} 个特征")
    print(f"原始X_pro shape: {datas['train_data']['X_pro'].shape}")
    print(f"原始X_pro_names: {len(datas['train_data']['X_pro_names'])} 个特征")

    # 添加Facebook粉丝数特征
    config = {}  # 空配置，因为函数内部会自动查找文件路径
    datas = add_facebook_fans_features(config, datas)

    print(f"\n更新后X_celeb shape: {datas['train_data']['X_celeb'].shape}")
    print(f"更新后X_celeb_names: {len(datas['train_data']['X_celeb_names'])} 个特征")
    print(f"更新后X_pro shape: {datas['train_data']['X_pro'].shape}")
    print(f"更新后X_pro_names: {len(datas['train_data']['X_pro_names'])} 个特征")

    # 打印统计信息
    print_facebook_feature_stats(datas)

    # 保存更新后的datas字典（覆盖原文件）
    print(f"\n保存更新后的datas字典到: {data_path}")
    with open(data_path, 'wb') as f:
        pickle.dump(datas, f)

    print("\n测试完成！Facebook粉丝数特征已添加到datas.pkl")


if __name__ == "__main__":
    main()
