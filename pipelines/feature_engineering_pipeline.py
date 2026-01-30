"""
特征工程主流程

整合所有特征工程模块
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config_loader import load_config
from src.data_manager import load_raw_data, save_data
from src.feature_engineering import (
    add_previous_seasons_count,
    add_gender_features,
    add_static_features,
    add_pro_features,
    add_dynamic_features,
    add_survival_features,
    add_context_features
)
from src.data_preprocessing import handle_missing_values, convert_to_long_format
from src.train_data_builder import build_train_data


def main():
    """特征工程主流程"""
    print("=" * 60)
    print("特征工程流程开始")
    print("=" * 60)

    # 加载配置
    config = load_config('configs/config.yaml')

    # 初始化数据字典
    datas = {}

    # ========== 阶段1：数据加载 ==========
    print("\n" + "=" * 60)
    print("阶段1：数据加载")
    print("=" * 60)
    datas = load_raw_data(config, datas)

    # ========== 阶段2：基础特征（宽格式） ==========
    print("\n" + "=" * 60)
    print("阶段2：基础特征构建（宽格式）")
    print("=" * 60)

    # 2.1 历史参赛次数
    print("\n[2.1] 添加历史参赛次数...")
    datas = add_previous_seasons_count(config, datas)

    # 2.2 性别识别
    print("\n[2.2] 添加性别特征...")
    datas = add_gender_features(config, datas)

    # ========== 阶段3：数据预处理 ==========
    print("\n" + "=" * 60)
    print("阶段3：数据预处理")
    print("=" * 60)
    datas = handle_missing_values(config, datas)

    # ========== 阶段4：静态特征 ==========
    print("\n" + "=" * 60)
    print("阶段4：静态特征构建")
    print("=" * 60)
    datas = add_static_features(config, datas)

    # 保存宽格式数据
    save_data(datas, 'data/processed/wide_format_with_features.pkl')
    print("\n宽格式数据已保存")

    # ========== 阶段5：格式转换 ==========
    print("\n" + "=" * 60)
    print("阶段5：格式转换（宽→长）")
    print("=" * 60)
    datas = convert_to_long_format(config, datas)

    # ========== 阶段6：职业舞者特征 ==========
    print("\n" + "=" * 60)
    print("阶段6：职业舞者特征")
    print("=" * 60)
    datas = add_pro_features(config, datas)

    # ========== 阶段7：动态特征 ==========
    print("\n" + "=" * 60)
    print("阶段7：动态特征")
    print("=" * 60)
    datas = add_dynamic_features(config, datas)

    # ========== 阶段8：历史轨迹特征 ==========
    print("\n" + "=" * 60)
    print("阶段8：历史轨迹特征")
    print("=" * 60)
    datas = add_survival_features(config, datas)

    # ========== 阶段9：赛制特征 ==========
    print("\n" + "=" * 60)
    print("阶段9：赛制特征")
    print("=" * 60)
    datas = add_context_features(config, datas)

    # ========== 阶段9.5：构建train_data格式 ==========
    datas = build_train_data(config, datas)

    # ========== 阶段10：保存最终数据 ==========
    print("\n" + "=" * 60)
    print("阶段10：保存最终数据")
    print("=" * 60)

    # 保存长格式数据
    save_data(datas, 'data/processed/long_format_with_all_features.pkl')

    # 导出CSV
    long_df = datas['long_data']
    output_csv = 'data/processed/long_format_features.csv'
    long_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"长格式CSV已保存到: {output_csv}")

    # 打印最终统计
    print("\n" + "=" * 60)
    print("特征工程完成统计")
    print("=" * 60)
    print(f"最终数据形状: {long_df.shape}")
    print(f"特征数量: {len(long_df.columns)}")
    print(f"\n特征列表:")
    for i, col in enumerate(long_df.columns, 1):
        print(f"  {i}. {col}")

    print("\n" + "=" * 60)
    print("特征工程流程完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
