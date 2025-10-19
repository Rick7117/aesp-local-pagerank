import random, os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings
import pandas as pd
import seaborn as sns
from collections import defaultdict

# 设置路径和参数
path = './datasets'
dataset = 'com-dblp'

eps = '1e-08'  # epsilon值
node_id = '17'  # 使用node_17的数据，也可以根据需要修改

# 设置matplotlib样式
warnings.filterwarnings("ignore")
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
sns.set_theme(style="white")
plt.rcParams['axes.labelsize'] = 25
plt.rcParams['xtick.labelsize'] = 20  
plt.rcParams['ytick.labelsize'] = 20

# 获取数据集中所有的alpha值
def get_all_alphas(dataset_path, method, node_id, eps):
    results_dir = os.path.join(dataset_path, 'results')
    alphas = set()
    
    if os.path.exists(results_dir):
        for file in os.listdir(results_dir):
            # 查找指定node_id的文件
            if file.startswith(f"{method}_node_{node_id}_alpha_") and f"_eps_{eps}" in file and file.endswith(".npz"):
                # 从文件名中提取alpha值
                alpha_str = file.split(f"{method}_node_{node_id}_alpha_")[1].split(f"_eps_{eps}")[0]
                try:
                    alpha = float(alpha_str)
                    alphas.add(alpha)
                except ValueError:
                    continue
    
    return sorted(list(alphas))

# 计算R值
def calculate_R(dataset_path, method, node_id, alphas, eps):
    R_values = []
    results_dir = os.path.join(dataset_path, 'results')
    
    for alpha in alphas:
        filename = f"{method}_node_{node_id}_alpha_{alpha}_eps_{eps}.npz"
        file_path = os.path.join(results_dir, filename)
        
        if os.path.exists(file_path):
            try:
                data = np.load(file_path, allow_pickle=True)
                grad_norms = [ii[0] for ii in data['grad_norms']]
                # 计算R值：梯度范数的最大值与初始梯度范数的比值
                r_value = np.max([val / grad_norms[0] for val in grad_norms])
                R_values.append(r_value)
                print(f"Alpha: {alpha}, R: {r_value}")
            except Exception as e:
                print(f"Error loading results for alpha={alpha}: {str(e)}")
                R_values.append(np.nan)  # 添加NaN表示缺失值
        else:
            print(f"File not found: {file_path}")
            R_values.append(np.nan)  # 添加NaN表示缺失值
    
    return R_values

# 主函数
def main():
    dataset_path = os.path.join(path, dataset)
    methods = ['aespappr', 'aesplocgd']
    
    # 用于保存CSV数据的列表
    csv_data = []
    
    # 创建第一个图表
    plt.figure(figsize=(6, 5))

    for method in methods:
        print(f"--- Processing method: {method} ---")
        # 获取所有alpha值
        alphas = get_all_alphas(dataset_path, method, node_id, eps)
        print(f"Found alphas: {alphas}")
        
        if not alphas:
            print(f"No alpha values found for {dataset} with method={method}, node_id={node_id}, eps={eps}")
            continue
        
        # 计算每个alpha对应的R值
        R_values = calculate_R(dataset_path, method, node_id, alphas, eps)
        
        # 收集CSV数据
        for alpha, r_value in zip(alphas, R_values):
            if not np.isnan(r_value):  # 只保存有效数据
                csv_data.append({
                    'method': method,
                    'alpha': alpha,
                    '1/alpha': 1/alpha,
                    'log(1/alpha)': np.log(1/alpha),
                    'R': r_value
                })
        
        # # 绘制R和alpha的关系图
        # 修改图例标签
        legend_label = 'AESP-LocAPPR' if method == 'aespappr' else 'AESP-LocGD' if method == 'aesplocgd' else method
        plt.plot(alphas, R_values, 'o-', linewidth=2, markersize=8, label=legend_label)
    
    # 设置第一个图表的坐标轴刻度
    plt.xscale('log')  # 保持x轴为对数刻度
    # 移除y轴对数刻度，直接显示R值
    
    # 添加标签和标题
    plt.xlabel(r'$\alpha$', fontsize=25)
    plt.ylabel(r'$R$', fontsize=25)
    plt.title(rf'Relationship between $R$ and $\alpha$ for {dataset}', fontsize=22)
    
    # 添加网格线和图例
    plt.grid(linestyle='-.', which='major')
    plt.legend(fontsize=18, loc='upper right')
    
    # 保存第一个图表
    os.makedirs('./figures', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'./figures/R_alpha_relationship_{dataset}.pdf', dpi=400, bbox_inches='tight')
    plt.savefig(f'./figures/R_alpha_relationship_{dataset}.png', dpi=400, bbox_inches='tight')
    print(f"Plot saved to ./figures/R_alpha_relationship_{dataset}.pdf and .png")

    # 创建第二个图表：R 和 log(1/alpha)
    plt.figure(figsize=(6, 5))
    for method in methods:
        alphas = get_all_alphas(dataset_path, method, node_id, eps)
        if not alphas:
            continue
        R_values = calculate_R(dataset_path, method, node_id, alphas, eps)
        
        # 计算 log(1/alpha)
        log_inv_alpha = np.log(1 / np.array(alphas))
        
        # 绘制R和log(1/alpha)的关系图
        # 修改图例标签
        legend_label = 'AESP-LocAPPR' if method == 'aespappr' else 'AESP-LocGD' if method == 'aesplocgd' else method
        plt.plot(log_inv_alpha, R_values, 'o-', linewidth=2, markersize=8, label=legend_label)

    # 添加标签和标题
    plt.xlabel(r'$\log(1/\alpha)$', fontsize=25)
    plt.ylabel(r'$R$', fontsize=25)
    plt.title(rf'Relationship between $R$ and $\log(1/\alpha)$ for {dataset}', fontsize=22)
    
    # 自定义对数坐标轴的格式，不使用科学计数法
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.2f}'.format(y)))
    
    # 添加网格线和图例
    plt.grid(linestyle='-.', which='major')
    plt.legend(fontsize=18, loc='upper left')
    
    # 保存第二个图表
    plt.tight_layout()
    plt.savefig(f'./figures/R_log_inv_alpha_relationship_{dataset}.pdf', dpi=400, bbox_inches='tight')
    plt.savefig(f'./figures/R_log_inv_alpha_relationship_{dataset}.png', dpi=400, bbox_inches='tight')
    print(f"Plot saved to ./figures/R_log_inv_alpha_relationship_{dataset}.pdf and .png")
    
    # 创建第三个图表：横向并列显示两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # 绘制第一个子图：R vs α
    for method in methods:
        alphas = get_all_alphas(dataset_path, method, node_id, eps)
        if not alphas:
            continue
        R_values = calculate_R(dataset_path, method, node_id, alphas, eps)
        
        # 修改图例标签
        legend_label = 'AESP-LocAPPR' if method == 'aespappr' else 'AESP-LocGD' if method == 'aesplocgd' else method
        ax1.plot(alphas, R_values, 'o-', linewidth=2, markersize=8, label=legend_label)
    
    # 设置第一个子图
    ax1.set_xscale('log')
    ax1.set_xlabel(r'$\alpha$', fontsize=25)
    ax1.set_ylabel(r'$R$', fontsize=25)
    ax1.set_title(rf'$R$ vs $\alpha$', fontsize=22)
    ax1.grid(linestyle='-.', which='major')
    ax1.legend(fontsize=18, loc='upper right')
    ax1.tick_params(axis='both', which='major', labelsize=20)
    
    # 绘制第二个子图：R vs log(1/α)
    for method in methods:
        alphas = get_all_alphas(dataset_path, method, node_id, eps)
        if not alphas:
            continue
        R_values = calculate_R(dataset_path, method, node_id, alphas, eps)
        
        # 计算 log(1/alpha)
        log_inv_alpha = np.log(1 / np.array(alphas))
        
        # 修改图例标签
        legend_label = 'AESP-LocAPPR' if method == 'aespappr' else 'AESP-LocGD' if method == 'aesplocgd' else method
        ax2.plot(log_inv_alpha, R_values, 'o-', linewidth=2, markersize=8, label=legend_label)
    
    # 设置第二个子图
    ax2.set_xlabel(r'$\log(1/\alpha)$', fontsize=25)
    ax2.set_ylabel(r'$R$', fontsize=25)
    ax2.set_title(rf'$R$ vs $\log(1/\alpha)$', fontsize=22)
    ax2.grid(linestyle='-.', which='major')
    ax2.legend(fontsize=18, loc='upper left')
    ax2.tick_params(axis='both', which='major', labelsize=20)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.2f}'.format(y)))
    
    # 调整子图间距
    plt.tight_layout()
    
    # 保存并列图表
    plt.savefig(f'./figures/R_alpha_combined_{dataset}.pdf', dpi=400, bbox_inches='tight')
    plt.savefig(f'./figures/R_alpha_combined_{dataset}.png', dpi=400, bbox_inches='tight')
    print(f"Combined plot saved to ./figures/R_alpha_combined_{dataset}.pdf and .png")
    
    # 保存CSV表格
    if csv_data:
        # 创建DataFrame
        df = pd.DataFrame(csv_data)
        
        # 创建tables目录
        tables_dir = '/mnt/data/binbin/git/NIPS2025/tables'
        os.makedirs(tables_dir, exist_ok=True)
        
        # 保存CSV文件
        csv_filename = os.path.join(tables_dir, f'R_alpha_results_{dataset}.csv')
        df.to_csv(csv_filename, index=False, float_format='%.6f')
        print(f"CSV table saved to {csv_filename}")
        
        # 显示前几行数据
        print("\nFirst 5 rows of the data:")
        print(df.head())
    else:
        print("No valid data found for CSV export.")

if __name__ == "__main__":
    main()