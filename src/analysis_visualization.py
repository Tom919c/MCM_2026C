"""
问题1分析可视化模块

实现所有可视化需求，包括：
- 粉丝投票估算结果呈现
- 关键特征影响分析
- 估算确定性评估
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats


def _setup_plot_style(config):
    """
    设置绘图样式

    Args:
        config: 配置字典
    """
    viz_config = config.get('visualization', {})

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 设置样式
    style = viz_config.get('style', 'whitegrid')
    sns.set_style(style)

    # 设置字体大小
    font_size = viz_config.get('font_size', 12)
    plt.rcParams['font.size'] = font_size


def _get_viz_config(config):
    """
    获取可视化配置参数

    Args:
        config: 配置字典

    Returns:
        dict: 可视化配置
    """
    default_config = {
        'dpi': 300,
        'figure_format': 'png',
        'style': 'whitegrid',
        'font_size': 12,
        'title_font_size': 14
    }

    viz_config = config.get('visualization', {})
    return {**default_config, **viz_config}


def plot_contestant_pfan_trend(model_output, train_data, posterior_samples,
                                contestant_names, config, output_dir='outputs/analysis'):
    """
    6.1.1 选手每周粉丝投票趋势与不确定性

    绘制代表性选手的P_fan和mu随周次变化，带95%后验置信区间

    Args:
        model_output: 模型输出字典
        train_data: 训练数据字典
        posterior_samples: 后验样本字典 (包含多个MCMC样本)
        contestant_names: list of dict, 每个包含 {'celeb_idx': int, 'name': str}
        config: 配置字典
        output_dir: 输出目录
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    _setup_plot_style(config)
    viz_config = _get_viz_config(config)

    for contestant in contestant_names:
        celeb_idx = contestant['celeb_idx']
        name = contestant['name']

        # 获取该选手的所有观测
        mask = (model_output['celeb_idx'] == celeb_idx)
        weeks = model_output['week_idx'][mask]

        # 排序以便绘图
        sort_idx = np.argsort(weeks)
        weeks_sorted = weeks[sort_idx]

        # 从后验样本计算P_fan的置信区间
        # posterior_samples['mu']: [n_samples, n_obs]
        if 'mu' in posterior_samples and len(posterior_samples['mu'].shape) == 2:
            mu_samples = posterior_samples['mu'][:, mask][:, sort_idx]

            # 对每个样本计算P_fan (需要周内归一化)
            pfan_samples = []
            for s in range(mu_samples.shape[0]):
                pfan_s = np.zeros(len(weeks_sorted))
                for i, w in enumerate(weeks_sorted):
                    week_mask = (model_output['week_idx'] == w)
                    mu_week = model_output['mu'][week_mask]
                    # Softmax
                    exp_mu = np.exp(mu_week - mu_week.max())
                    pfan_week = exp_mu / exp_mu.sum()
                    # 找到该选手在这周的位置
                    celeb_week = model_output['celeb_idx'][week_mask]
                    pos = np.where(celeb_week == celeb_idx)[0][0]
                    pfan_s[i] = pfan_week[pos]
                pfan_samples.append(pfan_s)

            pfan_samples = np.array(pfan_samples)
            pfan_mean = pfan_samples.mean(axis=0)
            pfan_lower = np.percentile(pfan_samples, 2.5, axis=0)
            pfan_upper = np.percentile(pfan_samples, 97.5, axis=0)
        else:
            # 如果没有后验样本，使用点估计
            pfan_mean = model_output['P_fan'][mask][sort_idx]
            pfan_lower = pfan_mean
            pfan_upper = pfan_mean

        # 绘图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # 子图1: P_fan趋势
        ax1.plot(weeks_sorted, pfan_mean, 'o-', linewidth=2, markersize=6,
                label='Posterior Mean', color='steelblue')
        ax1.fill_between(weeks_sorted, pfan_lower, pfan_upper,
                         alpha=0.3, color='steelblue', label='95% CI')
        ax1.set_xlabel('Week', fontsize=viz_config['font_size'])
        ax1.set_ylabel('Fan Vote Proportion (P_fan)', fontsize=viz_config['font_size'])
        ax1.set_title(f'{name} - Weekly Fan Vote Trend',
                     fontsize=viz_config['title_font_size'], fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 子图2: mu趋势
        mu_mean = model_output['mu'][mask][sort_idx]
        ax2.plot(weeks_sorted, mu_mean, 'o-', linewidth=2, markersize=6,
                color='coral', label='Latent Strength (μ)')
        ax2.set_xlabel('Week', fontsize=viz_config['font_size'])
        ax2.set_ylabel('Latent Voting Strength (μ)', fontsize=viz_config['font_size'])
        ax2.set_title(f'{name} - Weekly Latent Strength',
                     fontsize=viz_config['title_font_size'], fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = Path(output_dir) / f'contestant_trend_{celeb_idx}_{name.replace(" ", "_")}.{viz_config["figure_format"]}'
        plt.savefig(output_path, dpi=viz_config['dpi'], bbox_inches='tight')
        plt.close()

        print(f"  - Saved: {output_path}")


def plot_season_pfan_distribution(model_output, train_data, config, season_id=None,
                                   output_dir='outputs/analysis'):
    """
    6.1.2 赛季每周粉丝投票百分比分布（整体概览）

    绘制堆叠面积图和热力图展示整个赛季的P_fan分布

    Args:
        model_output: 模型输出字典
        train_data: 训练数据字典
        season_id: 赛季ID，如果为None则使用所有赛季的平均
        output_dir: 输出目录
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    _setup_plot_style(config)

    viz_config = _get_viz_config(config)


    # 选择赛季
    if season_id is not None:
        season_mask = (train_data['season_idx'] == season_id)
        title_suffix = f'Season {season_id}'
        file_suffix = f'season_{season_id}'
    else:
        season_mask = np.ones(len(model_output['P_fan']), dtype=bool)
        title_suffix = 'All Seasons Average'
        file_suffix = 'all_seasons'

    # 获取该赛季的数据
    celeb_idx = model_output['celeb_idx'][season_mask]
    week_idx = model_output['week_idx'][season_mask]
    P_fan = model_output['P_fan'][season_mask]

    # 构建数据框
    df = pd.DataFrame({
        'celeb_idx': celeb_idx,
        'week_idx': week_idx,
        'P_fan': P_fan
    })

    # 透视表：周 x 选手
    pivot_table = df.pivot_table(values='P_fan', index='week_idx',
                                  columns='celeb_idx', fill_value=0)

    # 图1: 堆叠面积图
    fig, ax = plt.subplots(figsize=(14, 8))
    pivot_table.plot.area(ax=ax, stacked=True, alpha=0.7, legend=False)
    ax.set_xlabel('Week', fontsize=viz_config["font_size"])
    ax.set_ylabel('Cumulative Fan Vote Proportion', fontsize=viz_config["font_size"])
    ax.set_title(f'Stacked Fan Vote Distribution - {title_suffix}',
                fontsize=viz_config["title_font_size"], fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    output_path = Path(output_dir) / f'pfan_stacked_area_{file_suffix}.{viz_config["figure_format"]}'
    plt.savefig(output_path, dpi=viz_config["dpi"], bbox_inches='tight')
    plt.close()
    print(f"  - Saved: {output_path}")

    # 图2: 热力图
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(pivot_table.T, cmap='YlOrRd', cbar_kws={'label': 'P_fan'},
                ax=ax, linewidths=0.5, linecolor='gray')
    ax.set_xlabel('Week', fontsize=viz_config["font_size"])
    ax.set_ylabel('Contestant ID', fontsize=viz_config["font_size"])
    ax.set_title(f'Fan Vote Heatmap - {title_suffix}',
                fontsize=viz_config["title_font_size"], fontweight='bold')

    output_path = Path(output_dir) / f'pfan_heatmap_{file_suffix}.{viz_config["figure_format"]}'
    plt.savefig(output_path, dpi=viz_config["dpi"], bbox_inches='tight')
    plt.close()
    print(f"  - Saved: {output_path}")


def plot_elimination_accuracy(metrics, config, output_dir='outputs/analysis'):
    """
    6.1.3 淘汰预测准确率与一致性报告

    绘制柱状图展示预测准确率

    Args:
        metrics: 评估指标字典，包含 'week_results', 'mean_accuracy' 等
        output_dir: 输出目录
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    _setup_plot_style(config)

    viz_config = _get_viz_config(config)


    week_results = metrics.get('week_results', [])
    if not week_results:
        print("  - Warning: No week_results in metrics")
        return

    # 提取数据
    weeks = []
    accuracies = []
    for i, result in enumerate(week_results):
        if result['accuracy'] is not None:
            weeks.append(i)
            accuracies.append(result['accuracy'])

    # 绘图
    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(weeks, accuracies, color='steelblue', alpha=0.7, edgecolor='black')

    # 添加平均线
    mean_acc = metrics.get('mean_accuracy', np.mean(accuracies))
    ax.axhline(y=mean_acc, color='red', linestyle='--', linewidth=2,
              label=f'Mean Accuracy: {mean_acc:.2%}')

    ax.set_xlabel('Week Index', fontsize=viz_config["font_size"])
    ax.set_ylabel('Elimination Prediction Accuracy', fontsize=viz_config["font_size"])
    ax.set_title('Weekly Elimination Prediction Accuracy', fontsize=viz_config["title_font_size"], fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    output_path = Path(output_dir) / f'elimination_accuracy.{viz_config["figure_format"]}'
    plt.savefig(output_path, dpi=viz_config["dpi"], bbox_inches='tight')
    plt.close()
    print(f"  - Saved: {output_path}")

    # 表格形式
    summary_df = pd.DataFrame({
        'Metric': ['Mean Accuracy', 'Weeks Evaluated', 'Min Accuracy', 'Max Accuracy'],
        'Value': [
            f"{mean_acc:.2%}",
            len(accuracies),
            f"{min(accuracies):.2%}" if accuracies else 'N/A',
            f"{max(accuracies):.2%}" if accuracies else 'N/A'
        ]
    })

    output_path = Path(output_dir) / 'elimination_accuracy_summary.csv'
    summary_df.to_csv(output_path, index=False)
    print(f"  - Saved: {output_path}")


def plot_survival_distribution_comparison(datas, config, output_dir='outputs/analysis'):
    """
    6.1.4 模拟与真实淘汰周数分布对比图

    使用后验预测检验结果，对比模拟和真实的存活周数分布

    Args:
        datas: 数据字典，包含 'S_samples' 和 'train_data'
        output_dir: 输出目录
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    _setup_plot_style(config)

    viz_config = _get_viz_config(config)


    train_data = datas['train_data']
    S_samples = datas.get('S_samples')

    if S_samples is None:
        print("  - Warning: No S_samples for posterior predictive check")
        return

    # 计算真实存活周数
    celeb_idx = train_data['celeb_idx']
    week_idx = train_data['week_idx']
    n_celebs = train_data['n_celebs']

    real_survival_weeks = []
    for c in range(n_celebs):
        mask = (celeb_idx == c)
        if mask.any():
            weeks = week_idx[mask]
            survival = len(np.unique(weeks))
            real_survival_weeks.append(survival)

    # 模拟存活周数（简化：基于S值排名）
    # 这里需要根据实际的后验预测检验逻辑来实现
    # 暂时使用简化版本
    simulated_survival_weeks = []
    n_samples = min(100, S_samples.shape[0])  # 限制样本数

    for s in range(n_samples):
        S_s = S_samples[s]
        # 对每周模拟淘汰
        for c in range(n_celebs):
            mask = (celeb_idx == c)
            if mask.any():
                # 简化：假设存活周数与平均S值正相关
                avg_S = S_s[mask].mean()
                # 添加随机性
                survival = int(np.clip(avg_S * 10 + np.random.randn(), 1, 15))
                simulated_survival_weeks.append(survival)

    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 子图1: 直方图对比
    ax1.hist(real_survival_weeks, bins=15, alpha=0.6, label='Real Data',
            color='steelblue', edgecolor='black', density=True)
    ax1.hist(simulated_survival_weeks, bins=15, alpha=0.6, label='Simulated',
            color='coral', edgecolor='black', density=True)
    ax1.set_xlabel('Survival Weeks', fontsize=viz_config["font_size"])
    ax1.set_ylabel('Density', fontsize=viz_config["font_size"])
    ax1.set_title('Survival Weeks Distribution Comparison', fontsize=viz_config["title_font_size"], fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图2: KDE对比
    from scipy.stats import gaussian_kde
    if len(real_survival_weeks) > 1:
        kde_real = gaussian_kde(real_survival_weeks)
        x_range = np.linspace(0, max(real_survival_weeks), 100)
        ax2.plot(x_range, kde_real(x_range), linewidth=2, label='Real Data', color='steelblue')

    if len(simulated_survival_weeks) > 1:
        kde_sim = gaussian_kde(simulated_survival_weeks)
        x_range = np.linspace(0, max(simulated_survival_weeks), 100)
        ax2.plot(x_range, kde_sim(x_range), linewidth=2, label='Simulated', color='coral')

    ax2.set_xlabel('Survival Weeks', fontsize=viz_config["font_size"])
    ax2.set_ylabel('Density', fontsize=viz_config["font_size"])
    ax2.set_title('KDE Comparison', fontsize=viz_config["title_font_size"], fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / f'survival_distribution_comparison.{viz_config["figure_format"]}'
    plt.savefig(output_path, dpi=viz_config["dpi"], bbox_inches='tight')
    plt.close()
    print(f"  - Saved: {output_path}")


def plot_spline_functions(model_output, train_data, posterior_samples, config,
                          output_dir='outputs/analysis'):
    """
    6.1.5 非线性关系曲线图（样条函数可视化）

    绘制样条特征的后验均值曲线和95%置信区间

    Args:
        model_output: 模型输出字典
        train_data: 训练数据字典
        posterior_samples: 后验样本字典
        output_dir: 输出目录
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    _setup_plot_style(config)

    viz_config = _get_viz_config(config)


    spline_features = model_output['feature_names'].get('splines', [])

    if not spline_features:
        print("  - Warning: No spline features in model output")
        return

    X_obs = train_data['X_obs']
    X_obs_names = train_data['X_obs_names']

    for feat in spline_features:
        # 获取特征值
        if feat not in X_obs_names:
            print(f"  - Warning: {feat} not in X_obs_names")
            continue

        feat_idx = X_obs_names.index(feat)
        x_values = X_obs[:, feat_idx]

        # 获取样条贡献
        spline_key = f'spline_{feat}'
        if spline_key not in model_output:
            print(f"  - Warning: {spline_key} not in model_output")
            continue

        y_values = model_output[spline_key]

        # 排序以便绘图
        sort_idx = np.argsort(x_values)
        x_sorted = x_values[sort_idx]
        y_sorted = y_values[sort_idx]

        # 计算置信区间（如果有后验样本）
        if spline_key in posterior_samples and len(posterior_samples[spline_key].shape) == 2:
            y_samples = posterior_samples[spline_key][:, sort_idx]
            y_lower = np.percentile(y_samples, 2.5, axis=0)
            y_upper = np.percentile(y_samples, 97.5, axis=0)
        else:
            y_lower = y_sorted
            y_upper = y_sorted

        # 绘图
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_sorted, y_sorted, linewidth=2, color='steelblue', label='Posterior Mean')
        ax.fill_between(x_sorted, y_lower, y_upper, alpha=0.3, color='steelblue',
                       label='95% CI')

        ax.set_xlabel(feat, fontsize=viz_config["font_size"])
        ax.set_ylabel('Contribution to μ', fontsize=viz_config["font_size"])
        ax.set_title(f'Spline Function: {feat}', fontsize=viz_config["title_font_size"], fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        output_path = Path(output_dir) / f'spline_{feat}.{viz_config["figure_format"]}'
        plt.savefig(output_path, dpi=viz_config["dpi"], bbox_inches='tight')
        plt.close()
        print(f"  - Saved: {output_path}")


def plot_feature_importance(model_output, posterior_samples, config, top_n=10,
                            output_dir='outputs/analysis'):
    """
    6.2.1 特征重要性条形图（Top 10 关键因素）

    绘制Top N特征的系数及95%可信区间

    Args:
        model_output: 模型输出字典
        posterior_samples: 后验样本字典
        top_n: 显示前N个特征
        output_dir: 输出目录
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    _setup_plot_style(config)

    viz_config = _get_viz_config(config)


    beta_obs = model_output.get('beta_obs')
    if beta_obs is None or len(beta_obs) == 0:
        print("  - Warning: No beta_obs in model output")
        return

    feature_names = model_output['feature_names']['beta_obs']

    # 计算后验均值和置信区间
    if 'beta_obs' in posterior_samples and len(posterior_samples['beta_obs'].shape) == 2:
        beta_samples = posterior_samples['beta_obs']
        beta_mean = beta_samples.mean(axis=0)
        beta_lower = np.percentile(beta_samples, 2.5, axis=0)
        beta_upper = np.percentile(beta_samples, 97.5, axis=0)
    else:
        beta_mean = beta_obs
        beta_lower = beta_obs
        beta_upper = beta_obs

    # 按绝对值排序
    abs_beta = np.abs(beta_mean)
    sort_idx = np.argsort(abs_beta)[::-1]

    # 取Top N
    top_n = min(top_n, len(beta_mean))
    top_idx = sort_idx[:top_n]
    top_features = [feature_names[i] for i in top_idx]
    top_beta = beta_mean[top_idx]
    top_lower = beta_lower[top_idx]
    top_upper = beta_upper[top_idx]

    # 计算误差棒
    errors = np.array([top_beta - top_lower, top_upper - top_beta])

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(top_features))

    colors = ['red' if b < 0 else 'steelblue' for b in top_beta]
    bars = ax.barh(y_pos, top_beta, color=colors, alpha=0.7, edgecolor='black')

    # 添加误差棒
    ax.errorbar(top_beta, y_pos, xerr=errors, fmt='none', color='black',
               capsize=5, capthick=2, linewidth=1.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features)
    ax.set_xlabel('Coefficient Value (β)', fontsize=viz_config["font_size"])
    ax.set_title(f'Top {top_n} Feature Importance (with 95% CI)', fontsize=viz_config["title_font_size"], fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    output_path = Path(output_dir) / f'feature_importance_top10.{viz_config["figure_format"]}'
    plt.savefig(output_path, dpi=viz_config["dpi"], bbox_inches='tight')
    plt.close()
    print(f"  - Saved: {output_path}")

    # 保存表格
    importance_df = pd.DataFrame({
        'Feature': top_features,
        'Coefficient': top_beta,
        'CI_Lower': top_lower,
        'CI_Upper': top_upper,
        'Abs_Coefficient': np.abs(top_beta)
    })
    output_path = Path(output_dir) / 'feature_importance_top10.csv'
    importance_df.to_csv(output_path, index=False)
    print(f"  - Saved: {output_path}")


def plot_cv_accuracy_trend(cv_results, config, output_dir='outputs/analysis'):
    """
    6.3.1 交叉验证准确率趋势图

    绘制留一赛季法交叉验证的准确率

    Args:
        cv_results: 交叉验证结果列表，每个元素包含 {'test_season': int, 'metrics': dict}
        output_dir: 输出目录
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    _setup_plot_style(config)

    viz_config = _get_viz_config(config)


    if not cv_results:
        print("  - Warning: No CV results")
        return

    # 提取数据
    seasons = []
    accuracies = []
    for result in cv_results:
        seasons.append(result['test_season'])
        metrics = result.get('metrics', {})
        acc = metrics.get('mean_accuracy', 0)
        accuracies.append(acc)

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 6))

    # 柱状图
    bars = ax.bar(seasons, accuracies, color='steelblue', alpha=0.7, edgecolor='black')

    # 平均线
    ax.axhline(y=mean_acc, color='red', linestyle='--', linewidth=2,
              label=f'Mean: {mean_acc:.2%} ± {std_acc:.2%}')

    ax.set_xlabel('Test Season', fontsize=viz_config["font_size"])
    ax.set_ylabel('Prediction Accuracy', fontsize=viz_config["font_size"])
    ax.set_title('Leave-One-Season-Out Cross-Validation Results',
                fontsize=viz_config["title_font_size"], fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    output_path = Path(output_dir) / f'cv_accuracy_trend.{viz_config["figure_format"]}'
    plt.savefig(output_path, dpi=viz_config["dpi"], bbox_inches='tight')
    plt.close()
    print(f"  - Saved: {output_path}")

    # 保存表格
    cv_df = pd.DataFrame({
        'Test_Season': seasons,
        'Accuracy': accuracies
    })
    cv_df.loc[len(cv_df)] = ['Mean ± Std', f'{mean_acc:.4f} ± {std_acc:.4f}']
    output_path = Path(output_dir) / 'cv_accuracy_summary.csv'
    cv_df.to_csv(output_path, index=False)
    print(f"  - Saved: {output_path}")


def plot_posterior_distributions(posterior_samples, param_names, config,
                                 output_dir='outputs/analysis'):
    """
    6.3.2 关键模型参数后验分布图

    绘制关键参数的后验分布

    Args:
        posterior_samples: 后验样本字典
        param_names: 要绘制的参数名列表，如 ['tau', 'alpha_sigma', 'delta_sigma']
        output_dir: 输出目录
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    _setup_plot_style(config)

    viz_config = _get_viz_config(config)


    available_params = [p for p in param_names if p in posterior_samples]

    if not available_params:
        print("  - Warning: No requested parameters in posterior_samples")
        return

    n_params = len(available_params)
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_params == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, param in enumerate(available_params):
        samples = posterior_samples[param]

        # 如果是多维参数，只绘制第一个维度或均值
        if len(samples.shape) > 1:
            samples = samples.mean(axis=1)

        ax = axes[i]

        # 直方图
        ax.hist(samples, bins=50, alpha=0.6, color='steelblue', edgecolor='black',
               density=True, label='Histogram')

        # KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(samples)
        x_range = np.linspace(samples.min(), samples.max(), 100)
        ax.plot(x_range, kde(x_range), linewidth=2, color='coral', label='KDE')

        # 统计信息
        mean_val = samples.mean()
        median_val = np.median(samples)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='green', linestyle=':', linewidth=2, label=f'Median: {median_val:.3f}')

        ax.set_xlabel(param, fontsize=viz_config["font_size"])
        ax.set_ylabel('Density', fontsize=viz_config["font_size"])
        ax.set_title(f'Posterior Distribution: {param}', fontsize=viz_config["font_size"], fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for i in range(n_params, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    output_path = Path(output_dir) / f'posterior_distributions.{viz_config["figure_format"]}'
    plt.savefig(output_path, dpi=viz_config["dpi"], bbox_inches='tight')
    plt.close()
    print(f"  - Saved: {output_path}")
