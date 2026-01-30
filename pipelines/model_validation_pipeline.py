"""
模型验证流程

加载训练好的模型，计算评估指标，生成可视化
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config_loader import load_config
from src.data_manager import load_data, save_data
from src.model_validation import (
    load_model_output,
    create_mock_model_output,
    calculate_accuracy,
    calculate_brier_score,
    plot_spline_functions,
    plot_feature_importance,
    plot_cross_validation_results
)


def main():
    """模型验证主流程"""
    print("=" * 60)
    print("模型验证流程开始")
    print("=" * 60)

    # 加载配置
    config = load_config('configs/config.yaml')

    # 加载训练数据
    print("\n加载训练数据...")
    datas = load_data('data/processed/datas.pkl')

    if 'train_data' not in datas:
        print("错误：数据字典中不存在train_data！")
        return

    train_data = datas['train_data']

    # ========== 加载或创建模型输出 ==========
    model_path = config.get('model_path', 'data/models/model.pkl')

    if Path(model_path).exists():
        print(f"\n加载模型输出: {model_path}")
        model_output = load_model_output(model_path)
    else:
        print(f"\n模型文件不存在: {model_path}")
        print("创建模拟模型输出用于测试...")
        model_output = create_mock_model_output(train_data)

    # 添加到数据字典
    datas['model_output'] = model_output

    # ========== 计算评估指标 ==========
    print("\n" + "=" * 60)
    print("计算评估指标")
    print("=" * 60)

    # 准确率
    accuracy, correct_weeks, total_weeks = calculate_accuracy(model_output, train_data)

    # Brier Score
    brier_score = calculate_brier_score(model_output, train_data)

    # ========== 生成可视化 ==========
    print("\n" + "=" * 60)
    print("生成可视化")
    print("=" * 60)

    output_dir = 'outputs/validation'

    # 样条函数曲线
    plot_spline_functions(model_output, train_data, output_dir)

    # 特征重要性
    plot_feature_importance(model_output, output_dir)

    # ========== 总结 ==========
    print("\n" + "=" * 60)
    print("验证结果总结")
    print("=" * 60)
    print(f"准确率: {accuracy:.2%} ({correct_weeks}/{total_weeks})")
    print(f"Brier Score: {brier_score:.4f}")
    print(f"\n可视化结果已保存到: {output_dir}")

    # ========== 保存结果 ==========
    print("\n" + "=" * 60)
    print("保存验证结果")
    print("=" * 60)

    # 保存包含model_output的数据字典
    save_data(datas, 'data/processed/validation_results.pkl')
    print("验证结果已保存到: data/processed/validation_results.pkl")

    # 如果使用的是模拟数据，额外保存模拟模型输出
    if not Path(model_path).exists():
        import pickle
        with open('data/models/mock_model_output.pkl', 'wb') as f:
            pickle.dump(model_output, f)
        print("模拟模型输出已保存到: data/models/mock_model_output.pkl")

    print("\n" + "=" * 60)
    print("模型验证流程完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
