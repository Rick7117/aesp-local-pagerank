import random, os
from utils import graph, read_files
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pandas as pd

sns.set_theme(style="white")
warnings.filterwarnings("ignore")
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"


def read_process():
    path = './datasets/com-dblp/results'
    methods = [
        'appr',
        'locgd',
        'aespappr',
        'aesplocgd'
    ]
    data_rows = []
    files = os.listdir(path)
    for file in files:
        params = file.split('_')
        method, node, alpha, eps = params[0], int(params[2]), float(params[4]), float(params[6].split('.npz')[0])
        if method not in methods:
            continue
        runtime = np.load(os.path.join(path, file), allow_pickle=True)['runtime']
        oper = sum(np.load(os.path.join(path, file), allow_pickle=True)['opers'])
        data_rows.append(
            {
                'method': method,
                'node': node,
                'alpha': alpha,
                'eps': eps,
                'runtime': runtime,
                'Opertion': oper
            }
        )
    data = pd.DataFrame(data_rows).drop_duplicates(subset=['method', 'node', 'alpha', 'eps'])
    data = data[data['eps'] == 1e-8]
    data.to_csv('./tables/runtime_operation_data.csv', index=False)

def speedup_process():
    data = pd.read_csv('./tables/runtime_operation_data.csv')

    grouped_data = data.drop_duplicates(subset=['method', 'node', 'alpha']).groupby(['node', 'alpha'])
    ratio_data = []

    for (node, alpha), group in grouped_data:
        if len(group) != 4:
            continue
        appr_data = group[group['method'] == 'appr'].iloc[0]
        locGD_data = group[group['method'] == 'locgd'].iloc[0]
        aesp_appr_data = group[group['method'] == 'aespappr'].iloc[0]
        aesp_locGD_data = group[group['method'] == 'aesplocgd'].iloc[0]
        ratio_data.append({
            'node': node,
            'alpha': alpha,
            'method': 'appr', 
            'runtime_ratio': appr_data['runtime'] / aesp_appr_data['runtime'],
            'operation_ratio': appr_data['Opertion'] / aesp_appr_data['Opertion']
        })
        ratio_data.append({
            'node': node,
            'alpha': alpha,
            'method': 'locgd',
            'runtime_ratio': locGD_data['runtime'] / aesp_locGD_data['runtime'],
            'operation_ratio': locGD_data['Opertion'] / aesp_locGD_data['Opertion']
        })
        
    ratio_data = pd.DataFrame(ratio_data)
    ratio_data.to_csv('./tables/runtime_operation_ratio.csv', index=False)

# read_process()
# speedup_process()
df = pd.read_csv('./tables/runtime_operation_ratio.csv')

alpha = [0.1, 0.01, 0.001]
indices = [1, 2, 3]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

box21 = ax1.boxplot(
    [df[(df['alpha'] == cat) & (df['method'] == 'appr')]['runtime_ratio'] for cat in df['alpha'].unique()],
    positions=np.arange(len(df['alpha'].unique())) - 0.2,  
    widths=0.3,
    patch_artist=True,
    boxprops=dict(facecolor='#1f77b4', alpha=0.7),
    medianprops=dict(color='#404040', linewidth=1, linestyle='-'),
    labels=df['alpha'].unique(),
    showfliers=False
)
box22 = ax1.boxplot(
    [df[(df['alpha'] == cat) & (df['method'] == 'locgd')]['runtime_ratio'] for cat in df['alpha'].unique()],
    positions=np.arange(len(df['alpha'].unique())) + 0.2,  
    widths=0.3,
    patch_artist=True,
    boxprops=dict(facecolor='#ff7f0e', alpha=0.7),
    medianprops=dict(color='#404040', linewidth=1, linestyle='-'),
    labels=df['alpha'].unique(),
    showfliers=False
)
ax1.set_ylabel('Speedup')
ax1.set_title('Running Time (s)')
ax1.tick_params(axis='y')
ax1.set_xlabel(r'$\alpha$')
ax1.set_xticks(np.arange(len(df['alpha'].unique())))
# ax1.set_xticklabels([r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$'])
ax1.set_xticklabels(df['alpha'].unique())
ax1.axhline(y=1, color='gray', linestyle='--', linewidth=1)

box11 = ax2.boxplot(
    [df[(df['alpha'] == cat) & (df['method'] == 'appr')]['operation_ratio'] for cat in df['alpha'].unique()],
    positions=np.arange(len(df['alpha'].unique())) - 0.2,  
    widths=0.3,
    patch_artist=True,
    boxprops=dict(facecolor='#1f77b4', alpha=0.7),
    medianprops=dict(color='#404040', linewidth=1, linestyle='-'),
    showfliers=False
)
box12 = ax2.boxplot(
    [df[(df['alpha'] == cat) & (df['method'] == 'locgd')]['operation_ratio'] for cat in df['alpha'].unique()],
    positions=np.arange(len(df['alpha'].unique())) + 0.2,  
    widths=0.3,
    patch_artist=True,
    boxprops=dict(facecolor='#ff7f0e', alpha=0.7),
    medianprops=dict(color='#404040', linewidth=1, linestyle='-'),
    showfliers=False
)

ax2.set_title(r'\# of Operations $(\sum_{t,k} \operatorname{vol}(\mathcal{S}_t^{(k)}))$')
ax2.tick_params(axis='y')
ax2.set_xlabel(r'$\alpha$')
ax2.set_xticks(np.arange(len(df['alpha'].unique())))
# ax2.set_xticklabels(df['alpha'].unique())
ax2.set_xticklabels([rf'$10^{{{int(i)}}}$' for i in np.log10(df['alpha'].unique())])
ax2.axhline(y=1, color='gray', linestyle='--', linewidth=1)

plt.legend(
    [box11['boxes'][0], box12['boxes'][0]], 
    ['APPR / AESP-LocAPPR', 'locGD / AESP-LocGD'], 
    loc='upper right'
)

plt.tight_layout()
plt.savefig('./figures/runtime_oper_accelerate.pdf', dpi=400, bbox_inches = 'tight')
