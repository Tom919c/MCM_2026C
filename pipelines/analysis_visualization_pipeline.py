"""
问题1分析可视化流程

加载模型输出和后验样本，生成所有可视化图表
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config_loader import load_config
from src.data_manager import load_data
from src.analysis_visualization import (
    plot_contestant_pfan_trend,
    plot_season_pfan_distribution,
    plot_elimination_accuracy,
    plot_survival_distribution_comparison,
    plot_spline_functions,
    plot_feature_importance,
    plot_cv_accuracy_trend,
    plot_posterior_distributions
)


def main():
    """分析可视化主流程"""
    print("=" * 60)
    print("问题1分析可视化流程开始")
    print("=" * 60)

    # 加载配置
    config = load_config('configs/config.yaml')

    # 加载训练结果
    print("\n加载训练结果...")
    datas = load_data('data/models/training_results.pkl')

    # 检查必需数据
    if 'model_output' not in datas:
        print("错误：training_results.pkl 中不存在 model_output")
        print("请先运行训练流程生成模型输出")
        return

    model_output = datas['model_output']
    train_data = datas['train_data']
    posterior_samples = datas.get('posterior_samples', {})
    metrics = datas.get('metrics', {})

    output_dir = 'outputs/analysis'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ========== 6.1 粉丝投票估算结果呈现 ==========
    print("\n" + "=" * 60)
    print("6.1 粉丝投票估算结果呈现")
    print("=" * 60)

    # 6.1.1 选手每周粉丝投票趋势
    print("\n[6.1.1] 绘制选手每周粉丝投票趋势...")
    # 示例：选择几位代表性选手
    # 需要根据实际数据确定选手ID和名称
    contestant_names = [
        {'celeb_idx': 0, 'name': 'Contestant_0'},
        {'celeb_idx': 1, 'name': 'Contestant_1'},
        # 可以添加更多选手
    ]
    plot_contestant_pfan_trend(model_output, train_data, posterior_samples,
                              contestant_names, config, output_dir)

    # 6.1.2 赛季每周粉丝投票百分比分布
    print("\n[6.1.2] 绘制赛季粉丝投票分布...")
    plot_season_pfan_distribution(model_output, train_data, config, season_id=0, output_dir=output_dir)
    plot_season_pfan_distribution(model_output, train_data, config, season_id=None, output_dir=output_dir)

    # 6.1.3 淘汰预测准确率
    print("\n[6.1.3] 绘制淘汰预测准确率...")
    if metrics:
        plot_elimination_accuracy(metrics, config, output_dir)
    else:
        print("  - 警告：无评估指标数据")

    # 6.1.4 模拟与真实淘汰周数分布对比
    print("\n[6.1.4] 绘制存活周数分布对比...")
    plot_survival_distribution_comparison(datas, config, output_dir)

    # 6.1.5 非线性关系曲线（样条函数）
    print("\n[6.1.5] 绘制样条函数曲线...")
    plot_spline_functions(model_output, train_data, posterior_samples, config, output_dir)

    # ========== 6.2 关键特征影响分析 ==========
    print("\n" + "=" * 60)
    print("6.2 关键特征影响分析")
    print("=" * 60)

    # 6.2.1 特征重要性Top 10
    print("\n[6.2.1] 绘制特征重要性...")
    plot_feature_importance(model_output, posterior_samples, config, top_n=10, output_dir=output_dir)

    # ========== 6.3 估算确定性评估 ==========
    print("\n" + "=" * 60)
    print("6.3 估算确定性评估")
    print("=" * 60)

    # 6.3.1 交叉验证准确率趋势
    print("\n[6.3.1] 绘制交叉验证准确率...")
    cv_results = datas.get('cv_results', [])
    if cv_results:
        plot_cv_accuracy_trend(cv_results, config, output_dir)
    else:
        print("  - 警告：无交叉验证结果")

    # 6.3.2 关键模型参数后验分布
    print("\n[6.3.2] 绘制参数后验分布...")
    param_names = ['tau', 'alpha_sigma', 'delta_sigma']
    plot_posterior_distributions(posterior_samples, param_names, config, output_dir)

    # ========== 完成 ==========
    print("\n" + "=" * 60)
    print("分析可视化流程完成")
    print("=" * 60)
    print(f"所有图表已保存到: {output_dir}")


if __name__ == "__main__":
    main()
