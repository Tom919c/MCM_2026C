"""
模型验证与诊断模块

实现模型输出加载、评估指标计算和可视化
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_model_output(model_path):
    """
    加载训练好的模型输出

    Args:
        model_path: 模型文件路径（.pkl）

    Returns:
        model_output: 符合MODEL_OUTPUT.md规范的字典
    """
    print(f"加载模型输出: {model_path}")

    with open(model_path, 'rb') as f:
        model_output = pickle.load(f)

    # 验证必需字段
    required_fields = [
        'n_obs', 'n_celebs', 'n_pros',
        'celeb_idx', 'pro_idx', 'week_idx',
        'mu', 'P_fan',
        'alpha_contrib', 'delta_contrib', 'linear_contrib', 'spline_contrib',
        'alpha', 'delta', 'beta_obs'
    ]

    for field in required_fields:
        if field not in model_output:
            raise ValueError(f"模型输出缺少必需字段: {field}")

    print(f"  - 观测数: {model_output['n_obs']}")
    print(f"  - 名人数: {model_output['n_celebs']}")
    print(f"  - 舞者数: {model_output['n_pros']}")

    return model_output


def create_mock_model_output(train_data):
    """
    创建模拟的模型输出用于测试

    基于train_data生成随机的模型输出

    Args:
        train_data: 训练数据字典

    Returns:
        model_output: 模拟的模型输出字典
    """
    print("创建模拟模型输出（用于测试）...")

    n_obs = train_data['n_obs']
    n_celebs = train_data['n_celebs']
    n_pros = train_data['n_pros']

    # 生成随机参数
    np.random.seed(42)
    alpha = np.random.randn(n_celebs) * 0.5
    delta = np.random.randn(n_pros) * 0.3
    beta_obs = np.random.randn(train_data['X_obs'].shape[1]) * 0.2

    # 计算mu（潜在投票强度）
    celeb_idx = train_data['celeb_idx']
    pro_idx = train_data['pro_idx']
    week_idx = train_data['week_idx']

    alpha_contrib = alpha[celeb_idx]
    delta_contrib = delta[pro_idx]
    linear_contrib = train_data['X_obs'] @ beta_obs
    spline_contrib = np.random.randn(n_obs) * 0.1  # 简化的样条贡献

    mu = alpha_contrib + delta_contrib + linear_contrib + spline_contrib

    # 计算P_fan（周内归一化）
    P_fan = np.zeros(n_obs)
    for w in range(train_data['n_weeks']):
        mask = (week_idx == w)
        if mask.any():
            mu_week = mu[mask]
            # Softmax归一化
            exp_mu = np.exp(mu_week - mu_week.max())
            P_fan[mask] = exp_mu / exp_mu.sum()

    model_output = {
        # 维度信息
        'n_obs': n_obs,
        'n_celebs': n_celebs,
        'n_pros': n_pros,

        # 索引数组
        'celeb_idx': celeb_idx,
        'pro_idx': pro_idx,
        'week_idx': week_idx,

        # 预测结果
        'mu': mu,
        'P_fan': P_fan,

        # 贡献分解
        'alpha_contrib': alpha_contrib,
        'delta_contrib': delta_contrib,
        'linear_contrib': linear_contrib,
        'spline_contrib': spline_contrib,

        # 模型参数
        'alpha': alpha,
        'delta': delta,
        'beta_obs': beta_obs,
        'spline_coefs': [],  # 简化

        # 特征名称
        'feature_names': {
            'beta_obs': train_data['X_obs_names'],
            'splines': []
        }
    }

    print(f"  - 模拟数据生成完成")

    return model_output


def calculate_accuracy(model_output, train_data):
    """
    计算预测准确率

    准确率 = 正确预测淘汰者的周数 / 总周数

    Args:
        model_output: 模型输出字典
        train_data: 训练数据字典（包含真实淘汰结果）

    Returns:
        accuracy: 准确率
        correct_weeks: 正确预测的周数
        total_weeks: 总周数
    """
    print("计算预测准确率...")

    P_fan = model_output['P_fan']
    week_idx = model_output['week_idx']
    n_weeks = train_data['n_weeks']
    week_data = train_data['week_data']

    correct_weeks = 0
    total_weeks = 0

    for w in range(n_weeks):
        week_info = week_data[w]

        # 跳过无淘汰的周
        if week_info['n_eliminated'] == 0:
            continue

        total_weeks += 1

        # 获取该周的观测
        mask = (week_idx == w)
        week_P_fan = P_fan[mask]
        week_eliminated = week_info['eliminated_mask'][mask]

        # 预测：P_fan最低的n_eliminated个选手会被淘汰
        n_eliminated = week_info['n_eliminated']
        predicted_eliminated_indices = np.argsort(week_P_fan)[:n_eliminated]

        # 真实淘汰的选手
        true_eliminated_indices = np.where(week_eliminated)[0]

        # 检查预测是否正确（集合相等）
        if set(predicted_eliminated_indices) == set(true_eliminated_indices):
            correct_weeks += 1

    accuracy = correct_weeks / total_weeks if total_weeks > 0 else 0

    print(f"  - 正确预测周数: {correct_weeks}/{total_weeks}")
    print(f"  - 准确率: {accuracy:.2%}")

    return accuracy, correct_weeks, total_weeks


def calculate_brier_score(model_output, train_data):
    """
    计算Brier Score

    Brier Score = (1/N) * Σ(p_i - y_i)^2
    其中p_i是预测概率，y_i是真实结果（0或1）

    Args:
        model_output: 模型输出字典
        train_data: 训练数据字典

    Returns:
        brier_score: Brier Score值（越低越好）
    """
    print("计算Brier Score...")

    # 将P_fan转换为淘汰概率
    # P_fan低 → 淘汰概率高
    P_fan = model_output['P_fan']
    week_idx = model_output['week_idx']
    n_weeks = train_data['n_weeks']
    week_data = train_data['week_data']

    # 计算每个观测的淘汰概率
    P_elimination = np.zeros(len(P_fan))

    for w in range(n_weeks):
        mask = (week_idx == w)
        week_P_fan = P_fan[mask]
        n_eliminated = week_data[w]['n_eliminated']

        if n_eliminated > 0 and mask.any():
            # 淘汰概率 = 1 - P_fan的归一化排名
            # P_fan最低的n_eliminated个选手淘汰概率最高
            ranks = np.argsort(np.argsort(week_P_fan))  # 0=最低P_fan
            n_contestants = len(week_P_fan)
            # 简化：线性映射
            P_elimination[mask] = 1 - (ranks / (n_contestants - 1))

    # 真实淘汰结果（0或1）
    y_true = np.zeros(len(P_fan))
    for w in range(n_weeks):
        mask = (week_idx == w)
        y_true[mask] = week_data[w]['eliminated_mask'][mask].astype(float)

    # 计算Brier Score
    brier_score = np.mean((P_elimination - y_true) ** 2)

    print(f"  - Brier Score: {brier_score:.4f}")

    return brier_score


def plot_spline_functions(model_output, train_data, output_dir='outputs/validation'):
    """
    绘制样条函数曲线

    为每个样条特征绘制后验均值曲线和95%置信区间

    Args:
        model_output: 模型输出字典
        train_data: 训练数据字典
        output_dir: 输出目录
    """
    print("绘制样条函数曲线...")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    spline_features = model_output['feature_names'].get('splines', [])

    if not spline_features:
        print("  - 警告：模型输出中没有样条特征")
        return

    for feat in spline_features:
        # 获取该特征的样条贡献
        contrib_key = f'spline_{feat}'
        if contrib_key not in model_output:
            print(f"  - 跳过 {feat}：未找到贡献数据")
            continue

        # 获取特征值和贡献
        feat_idx = train_data['X_obs_names'].index(feat)
        x_values = train_data['X_obs'][:, feat_idx]
        y_values = model_output[contrib_key]

        # 排序以便绘图
        sort_idx = np.argsort(x_values)
        x_sorted = x_values[sort_idx]
        y_sorted = y_values[sort_idx]

        # 绘图
        plt.figure(figsize=(10, 6))
        plt.plot(x_sorted, y_sorted, 'b-', linewidth=2, label='后验均值')

        # TODO: 添加95%置信区间（需要后验样本）
        # plt.fill_between(x_sorted, lower_ci, upper_ci, alpha=0.3, label='95% CI')

        plt.xlabel(feat, fontsize=12)
        plt.ylabel('贡献值', fontsize=12)
        plt.title(f'样条函数: {feat}', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        output_path = Path(output_dir) / f'spline_{feat}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  - 已保存: {output_path}")


def plot_feature_importance(model_output, output_dir='outputs/validation'):
    """
    绘制特征重要性图

    显示线性特征系数的后验均值和95%可信区间

    Args:
        model_output: 模型输出字典
        output_dir: 输出目录
    """
    print("绘制特征重要性图...")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    beta_obs = model_output['beta_obs']
    feature_names = model_output['feature_names']['beta_obs']

    # 按|β|排序
    abs_beta = np.abs(beta_obs)
    sort_idx = np.argsort(abs_beta)[::-1]

    # 取Top 10
    top_n = min(10, len(beta_obs))
    top_idx = sort_idx[:top_n]
    top_features = [feature_names[i] for i in top_idx]
    top_beta = beta_obs[top_idx]

    # 绘图
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(top_features))

    colors = ['red' if b < 0 else 'blue' for b in top_beta]
    plt.barh(y_pos, top_beta, color=colors, alpha=0.7)

    # TODO: 添加95%可信区间（需要后验样本）
    # plt.errorbar(top_beta, y_pos, xerr=ci_width, fmt='none', color='black', capsize=5)

    plt.yticks(y_pos, top_features)
    plt.xlabel('系数值 (β)', fontsize=12)
    plt.title('Top 10 特征重要性', fontsize=14)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.grid(True, alpha=0.3, axis='x')

    output_path = Path(output_dir) / 'feature_importance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  - 已保存: {output_path}")

    # 打印Top 10特征
    print("\nTop 10 特征:")
    for i, (feat, beta) in enumerate(zip(top_features, top_beta), 1):
        print(f"  {i}. {feat:30s}: β = {beta:7.4f}")


def plot_cross_validation_results(cv_results, output_dir='outputs/validation'):
    """
    绘制交叉验证结果

    显示各赛季的预测准确率

    Args:
        cv_results: 交叉验证结果字典
            {
                'season_ids': list,
                'accuracies': list,
                'mean_accuracy': float,
                'std_accuracy': float
            }
        output_dir: 输出目录
    """
    print("绘制交叉验证结果...")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    season_ids = cv_results['season_ids']
    accuracies = cv_results['accuracies']
    mean_acc = cv_results['mean_accuracy']
    std_acc = cv_results['std_accuracy']

    # 绘图
    plt.figure(figsize=(12, 6))

    # 条形图
    plt.bar(range(len(season_ids)), accuracies, alpha=0.7, color='steelblue')

    # 平均线
    plt.axhline(y=mean_acc, color='red', linestyle='--', linewidth=2,
                label=f'平均准确率: {mean_acc:.2%} ± {std_acc:.2%}')

    plt.xlabel('赛季', fontsize=12)
    plt.ylabel('准确率', fontsize=12)
    plt.title('留一赛季法交叉验证结果', fontsize=14)
    plt.xticks(range(len(season_ids)), season_ids, rotation=45)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    output_path = Path(output_dir) / 'cross_validation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  - 已保存: {output_path}")
    print(f"  - 平均准确率: {mean_acc:.2%} ± {std_acc:.2%}")

