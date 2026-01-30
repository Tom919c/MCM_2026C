"""
可视化模块测试脚本

创建模拟数据并测试所有可视化函数
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.config_loader import load_config
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


def create_mock_data():
    """创建模拟数据用于测试"""
    print("创建模拟数据...")

    # 维度
    n_obs = 100
    n_celebs = 10
    n_pros = 5
    n_weeks = 10
    n_seasons = 2

    # 模型输出
    model_output = {
        'n_obs': n_obs,
        'n_celebs': n_celebs,
        'n_pros': n_pros,

        'celeb_idx': np.random.randint(0, n_celebs, n_obs),
        'pro_idx': np.random.randint(0, n_pros, n_obs),
        'week_idx': np.random.randint(0, n_weeks, n_obs),

        'mu': np.random.randn(n_obs),
        'P_fan': np.random.uniform(0, 1, n_obs),

        'alpha_contrib': np.random.randn(n_obs),
        'delta_contrib': np.random.randn(n_obs),
        'linear_contrib': np.random.randn(n_obs),
        'spline_contrib': np.random.randn(n_obs),

        'alpha': np.random.randn(n_celebs),
        'delta': np.random.randn(n_pros),
        'beta_obs': np.random.randn(5),
        'spline_coefs': [],

        'feature_names': {
            'beta_obs': ['feature1', 'feature2', 'feature3', 'feature4', 'feature5'],
            'splines': ['z_score', 'weeks_survived']
        },

        # 添加样条特征贡献
        'spline_z_score': np.random.randn(n_obs) * 0.1,
        'spline_weeks_survived': np.random.randn(n_obs) * 0.1,
    }

    # 训练数据
    train_data = {
        'n_obs': n_obs,
        'n_weeks': n_weeks,
        'n_celebs': n_celebs,
        'n_pros': n_pros,
        'n_seasons': n_seasons,

        'celeb_idx': model_output['celeb_idx'],
        'pro_idx': model_output['pro_idx'],
        'week_idx': model_output['week_idx'],
        'season_idx': np.random.randint(0, n_seasons, n_obs),

        'X_celeb': np.random.randn(n_celebs, 3),
        'X_pro': np.random.randn(n_pros, 2),
        'X_obs': np.random.randn(n_obs, 5),

        'X_celeb_names': ['celeb_feat1', 'celeb_feat2', 'celeb_feat3'],
        'X_pro_names': ['pro_feat1', 'pro_feat2'],
        'X_obs_names': ['z_score', 'weeks_survived', 'obs_feat3', 'obs_feat4', 'obs_feat5'],

        'judge_score_pct': np.random.uniform(0, 1, n_obs),
        'judge_rank_score': np.random.uniform(0, 1, n_obs),

        'week_data': []
    }

    # 构建周级数据
    for w in range(n_weeks):
        mask = (train_data['week_idx'] == w)
        n_contestants = mask.sum()
        eliminated_mask = np.zeros(n_obs, dtype=bool)
        if n_contestants > 0:
            # 随机淘汰1人
            week_indices = np.where(mask)[0]
            if len(week_indices) > 0:
                elim_idx = np.random.choice(week_indices)
                eliminated_mask[elim_idx] = True

        train_data['week_data'].append({
            'obs_mask': mask,
            'n_contestants': int(n_contestants),
            'n_eliminated': int(eliminated_mask.sum()),
            'eliminated_mask': eliminated_mask,
            'rule_method': w % 2,  # 交替使用两种规则
            'judge_save_active': w > 5,
            'season': w // 5,
            'week': w % 5
        })

    # 后验样本（模拟MCMC样本）
    n_samples = 50
    posterior_samples = {
        'alpha': np.random.randn(n_samples, n_celebs),
        'delta': np.random.randn(n_samples, n_pros),
        'beta_obs': np.random.randn(n_samples, 5),
        'mu': np.random.randn(n_samples, n_obs),
        'tau': np.random.gamma(2, 1, n_samples),
        'alpha_sigma': np.random.gamma(2, 1, n_samples),
        'delta_sigma': np.random.gamma(2, 1, n_samples),

        # 样条特征后验
        'spline_z_score': np.random.randn(n_samples, n_obs) * 0.1,
        'spline_weeks_survived': np.random.randn(n_samples, n_obs) * 0.1,
    }

    # 评估指标
    metrics = {
        'week_results': [
            {'accuracy': np.random.uniform(0.5, 1.0), 'season': w // 5, 'week': w % 5}
            for w in range(n_weeks)
        ],
        'mean_accuracy': 0.75,
        'n_weeks_evaluated': n_weeks
    }

    # 交叉验证结果
    cv_results = [
        {
            'test_season': s,
            'metrics': {
                'mean_accuracy': np.random.uniform(0.6, 0.9),
                'n_weeks_evaluated': 5
            }
        }
        for s in range(n_seasons)
    ]

    # S_samples用于后验预测检验
    S_samples = np.random.randn(n_samples, n_obs)

    # 组装数据字典
    datas = {
        'model_output': model_output,
        'train_data': train_data,
        'posterior_samples': posterior_samples,
        'metrics': metrics,
        'cv_results': cv_results,
        'S_samples': S_samples
    }

    print("  - 模拟数据创建完成")
    return datas


def test_visualization_functions(config, datas):
    """测试所有可视化函数"""
    print("\n" + "=" * 60)
    print("测试可视化函数")
    print("=" * 60)

    model_output = datas['model_output']
    train_data = datas['train_data']
    posterior_samples = datas['posterior_samples']
    metrics = datas['metrics']
    cv_results = datas['cv_results']

    output_dir = 'outputs/test_visualization'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    test_results = []

    # 测试1: 选手每周粉丝投票趋势
    print("\n[Test 1/8] plot_contestant_pfan_trend...")
    try:
        contestant_names = [
            {'celeb_idx': 0, 'name': 'Test_Contestant_0'},
            {'celeb_idx': 1, 'name': 'Test_Contestant_1'},
        ]
        plot_contestant_pfan_trend(model_output, train_data, posterior_samples,
                                  contestant_names, config, output_dir)
        test_results.append(('plot_contestant_pfan_trend', 'PASS'))
        print("  [PASS]")
    except Exception as e:
        test_results.append(('plot_contestant_pfan_trend', f'FAIL: {str(e)}'))
        print(f"  [FAIL]: {e}")

    # 测试2: 赛季粉丝投票分布
    print("\n[Test 2/8] plot_season_pfan_distribution...")
    try:
        plot_season_pfan_distribution(model_output, train_data, config,
                                     season_id=0, output_dir=output_dir)
        test_results.append(('plot_season_pfan_distribution', 'PASS'))
        print("  ✓ PASS")
    except Exception as e:
        test_results.append(('plot_season_pfan_distribution', f'FAIL: {str(e)}'))
        print(f"  ✗ FAIL: {e}")

    # 测试3: 淘汰预测准确率
    print("\n[Test 3/8] plot_elimination_accuracy...")
    try:
        plot_elimination_accuracy(metrics, config, output_dir)
        test_results.append(('plot_elimination_accuracy', 'PASS'))
        print("  ✓ PASS")
    except Exception as e:
        test_results.append(('plot_elimination_accuracy', f'FAIL: {str(e)}'))
        print(f"  ✗ FAIL: {e}")

    # 测试4: 存活周数分布对比
    print("\n[Test 4/8] plot_survival_distribution_comparison...")
    try:
        plot_survival_distribution_comparison(datas, config, output_dir)
        test_results.append(('plot_survival_distribution_comparison', 'PASS'))
        print("  ✓ PASS")
    except Exception as e:
        test_results.append(('plot_survival_distribution_comparison', f'FAIL: {str(e)}'))
        print(f"  ✗ FAIL: {e}")

    # 测试5: 样条函数曲线
    print("\n[Test 5/8] plot_spline_functions...")
    try:
        plot_spline_functions(model_output, train_data, posterior_samples,
                             config, output_dir)
        test_results.append(('plot_spline_functions', 'PASS'))
        print("  ✓ PASS")
    except Exception as e:
        test_results.append(('plot_spline_functions', f'FAIL: {str(e)}'))
        print(f"  ✗ FAIL: {e}")

    # 测试6: 特征重要性
    print("\n[Test 6/8] plot_feature_importance...")
    try:
        plot_feature_importance(model_output, posterior_samples, config,
                               top_n=5, output_dir=output_dir)
        test_results.append(('plot_feature_importance', 'PASS'))
        print("  ✓ PASS")
    except Exception as e:
        test_results.append(('plot_feature_importance', f'FAIL: {str(e)}'))
        print(f"  ✗ FAIL: {e}")

    # 测试7: 交叉验证准确率
    print("\n[Test 7/8] plot_cv_accuracy_trend...")
    try:
        plot_cv_accuracy_trend(cv_results, config, output_dir)
        test_results.append(('plot_cv_accuracy_trend', 'PASS'))
        print("  ✓ PASS")
    except Exception as e:
        test_results.append(('plot_cv_accuracy_trend', f'FAIL: {str(e)}'))
        print(f"  ✗ FAIL: {e}")

    # 测试8: 参数后验分布
    print("\n[Test 8/8] plot_posterior_distributions...")
    try:
        param_names = ['tau', 'alpha_sigma', 'delta_sigma']
        plot_posterior_distributions(posterior_samples, param_names, config, output_dir)
        test_results.append(('plot_posterior_distributions', 'PASS'))
        print("  ✓ PASS")
    except Exception as e:
        test_results.append(('plot_posterior_distributions', f'FAIL: {str(e)}'))
        print(f"  ✗ FAIL: {e}")

    return test_results


def print_test_summary(test_results):
    """打印测试总结"""
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    passed = sum(1 for _, result in test_results if result == 'PASS')
    total = len(test_results)

    print(f"\n总计: {passed}/{total} 测试通过\n")

    for func_name, result in test_results:
        status = "[PASS]" if result == "PASS" else "[FAIL]"
        print(f"{status} {func_name}: {result}")

    if passed == total:
        print("\n所有测试通过！")
        return True
    else:
        print(f"\n{total - passed} 个测试失败")
        return False


def main():
    """主测试流程"""
    print("=" * 60)
    print("可视化模块测试")
    print("=" * 60)

    # 加载配置
    print("\n加载配置...")
    config = load_config('configs/config.yaml')
    print(f"  - DPI: {config['visualization']['dpi']}")
    print(f"  - Format: {config['visualization']['figure_format']}")

    # 创建模拟数据
    datas = create_mock_data()

    # 运行测试
    test_results = test_visualization_functions(config, datas)

    # 打印总结
    all_passed = print_test_summary(test_results)

    # 检查输出文件
    print("\n" + "=" * 60)
    print("检查输出文件")
    print("=" * 60)
    output_dir = Path('outputs/test_visualization')
    if output_dir.exists():
        files = list(output_dir.glob(f'*.{config["visualization"]["figure_format"]}'))
        print(f"\n生成了 {len(files)} 个图表文件:")
        for f in sorted(files):
            print(f"  - {f.name}")
    else:
        print("\n警告：输出目录不存在")

    print("\n" + "=" * 60)
    if all_passed:
        print("[PASS] 测试完成：所有功能正常")
    else:
        print("[FAIL] 测试完成：部分功能异常")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
