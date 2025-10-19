import random, os
from utils import graph, read_files
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pandas as pd


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

alphas = df['alpha'].unique()

sns.set_theme(style="white")
warnings.filterwarnings("ignore")
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18  
plt.rcParams['axes.labelsize'] = 23
plt.rcParams['axes.titlesize'] = 25

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

runtime_ratio_mean = [np.mean(df[(df['alpha'] == cat) & (df['method'] == 'appr')]['runtime_ratio']) for cat in alphas]
runtime_ratio_max = [np.max(df[(df['alpha'] == cat) & (df['method'] == 'appr')]['runtime_ratio']) for cat in alphas]
runtime_ratio_min = [np.min(df[(df['alpha'] == cat) & (df['method'] == 'appr')]['runtime_ratio']) for cat in alphas]
ax1.plot(alphas, runtime_ratio_mean, color='#1f77b4', label='APPR / AESP-LocAPPR')
ax1.fill_between(alphas, 
                 np.array(runtime_ratio_min),
                 np.array(runtime_ratio_max),
                 color='#1f77b4', alpha=0.2)
runtime_ratio_mean = [np.mean(df[(df['alpha'] == cat) & (df['method'] == 'locgd')]['runtime_ratio']) for cat in alphas]
runtime_ratio_max = [np.max(df[(df['alpha'] == cat) & (df['method'] == 'locgd')]['runtime_ratio']) for cat in alphas]
runtime_ratio_min = [np.min(df[(df['alpha'] == cat) & (df['method'] == 'locgd')]['runtime_ratio']) for cat in alphas]
ax1.plot(alphas, runtime_ratio_mean, color='#ff7f0e', label='locGD / AESP-LocGD')
ax1.fill_between(alphas, 
                 np.array(runtime_ratio_min),
                 np.array(runtime_ratio_max),
                 color='#ff7f0e', alpha=0.2)

ax1.set_ylabel('Speedup')
ax1.set_title('Running Time (s)', y=1.02)
ax1.tick_params(axis='y')
ax1.set_xlabel(r'$\alpha$')
# ax1.axhline(y=1, color='gray', linestyle='--', linewidth=1)
ax1.grid(linestyle='-.', which='major')

operation_ratio_mean = [np.mean(df[(df['alpha'] == cat) & (df['method'] == 'appr')]['operation_ratio']) for cat in alphas]
operation_ratio_max = [np.max(df[(df['alpha'] == cat) & (df['method'] == 'appr')]['operation_ratio']) for cat in alphas]
operation_ratio_min = [np.min(df[(df['alpha'] == cat) & (df['method'] == 'appr')]['operation_ratio']) for cat in alphas]
ax2.plot(alphas, operation_ratio_mean, color='#1f77b4', label='APPR / AESP-LocAPPR')
ax2.fill_between(alphas, 
                 np.array(operation_ratio_min),
                 np.array(operation_ratio_max),
                 color='#1f77b4', alpha=0.2)
operation_ratio_mean = [np.mean(df[(df['alpha'] == cat) & (df['method'] == 'locgd')]['operation_ratio']) for cat in alphas]
operation_ratio_max = [np.max(df[(df['alpha'] == cat) & (df['method'] == 'locgd')]['operation_ratio']) for cat in alphas]
operation_ratio_min = [np.min(df[(df['alpha'] == cat) & (df['method'] == 'locgd')]['operation_ratio']) for cat in alphas]
ax2.plot(alphas, operation_ratio_mean, color='#ff7f0e', label='LocGD / AESP-LocGD')
ax2.fill_between(alphas, 
                 np.array(operation_ratio_min),
                 np.array(operation_ratio_max),
                 color='#ff7f0e', alpha=0.2)

# ax2.set_title(r'\# of Operations $(\sum_{t,k} \operatorname{vol}(\mathcal{S}_t^{(k)}))$')
# ax2.axhline(y=1, color='gray', linestyle='--', linewidth=1)
ax2.set_title(r'\# of Operations $(\sum_{t,k} \operatorname{vol}(\mathcal{S}_t^{(k)}))$', 
              y=1.02)
ax2.set_xlabel(r'$\alpha$')
ax2.legend(fontsize=22)
ax2.grid(linestyle='-.', which='major')
plt.tight_layout()
plt.savefig('./figures/runtime_oper_accelerate_lineplot.pdf', dpi=400, bbox_inches = 'tight')
