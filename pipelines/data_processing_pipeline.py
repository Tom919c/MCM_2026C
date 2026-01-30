"""
数据处理流程

功能：
1. 加载原始数据
2. 添加历史参赛次数特征
3. 添加性别识别特征
4. 保存处理后的数据
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config_loader import load_config
from src.data_manager import load_raw_data, save_data
from src.feature_engineering import add_previous_seasons_count, add_gender_features


def main():
    """主流程"""
    print("=" * 60)
    print("数据处理流程开始")
    print("=" * 60)

    # 加载配置
    config = load_config('configs/config.yaml')

    # 初始化数据字典
    datas = {}

    # ========== 数据加载阶段 ==========
    print("\n[1/4] 加载原始数据")
    datas = load_raw_data(config, datas)

    # ========== 特征工程阶段 ==========
    print("\n[2/4] 添加历史参赛次数特征")
    datas = add_previous_seasons_count(config, datas)

    print("\n[3/4] 添加性别识别特征")
    datas = add_gender_features(config, datas)

    # ========== 数据保存阶段 ==========
    print("\n[4/4] 保存处理后的数据")

    # 注意：不再保存 data_with_features.pkl，请使用 feature_engineering_pipeline.py 生成完整数据
    # save_data(datas, 'data/processed/data_with_features.pkl')

    # 导出特征数据CSV（以行号为索引）
    output_features_csv = 'data/processed/features.csv'
    datas['features'].to_csv(output_features_csv, index=True, encoding='utf-8-sig')
    print(f"特征数据CSV已保存到: {output_features_csv}")

    print("\n" + "=" * 60)
    print("数据处理流程完成")
    print("=" * 60)



if __name__ == "__main__":
    main()
