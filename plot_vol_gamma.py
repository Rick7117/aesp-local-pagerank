import random, os
from utils import graph, read_files
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import warnings
import pandas as pd

path = './datasets'
sorted_graphs = [
    ("ogbn-arxiv", 169343, 1157799),
    ("ogbn-proteins", 132534, 39561252),
    ("com-dblp", 317080, 1049866),
    ("com-youtube", 1134890, 2987624),
    ("soc-pokec", 1632803, 22301964),
    ("wiki-talk", 2388953, 4656682),
    ("ogbl-ppa", 576039, 21231776),
    ("com-lj", 3997962, 34681189),
    ("ogbn-mag", 1939743, 21091072),
    ("soc-lj1", 4843953, 42845684),
    ("sx-stackoverflow", 2584164, 28183518),
    ("as-skitter", 1694616, 11094209),
    ("ogbn-products", 2385902, 61806303),
    ("com-orkut", 3072441, 117185083),
    ("cit-patent", 3764117, 16511740),
    ("wiki-en21", 6216199, 160823797),
    ("com-friendster", 65608366, 1806067135),
    ("ogbn-papers100M", 111059433, 1615685450),
    ("ogb-mag240m", 244160499, 1728364232)
]

datasets = [graph_info[0] for graph_info in sorted_graphs]
all_results = {}
alpha, eps = 0.1, 1e-06
method = "aespappr"

grad_ht_scale_all = []
vol_gamma_all = []

for dataset in datasets:
    dataset_path = os.path.join(path, dataset)
    if os.path.isdir(dataset_path):
        results_dir = os.path.join(dataset_path, 'results')
        if os.path.exists(results_dir):
            matching_files = [f for f in os.listdir(results_dir) 
                           if f.startswith(f"{method}_node_") and f.endswith(f"_alpha_{alpha}_eps_{eps}.npz")]      
            if matching_files:
                results_file = os.path.join(results_dir, matching_files[0])
                results_data = np.load(results_file, allow_pickle=True)
                grad_norms = [ii[0] for ii in results_data['grad_norms']]
                eps_t = results_data['eps_t']
                grad_norms_scale = [grad_norms[i] / eps_t[i] for i in range(len(grad_norms))]
                grad_ht_scale_all.append(grad_norms_scale)
                vols_bar = [np.mean(vol_st) for vol_st in results_data['vols']] 
                gammas_bar = [np.mean(gamma_t) for gamma_t in results_data['gammas']]
                vol_gamma = [vols_bar[i] / gammas_bar[i] for i in range(len(vols_bar))]
                vol_gamma_all.append(vol_gamma)
            
            else:
                print(f"No matching .npz file found for {dataset} (method={method}, alpha={alpha}, eps={eps})")
            
n_graphs = len(vol_gamma_all)
print(n_graphs)
indices = np.arange(n_graphs) + 1

sns.set_theme(style="white")
warnings.filterwarnings("ignore")
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
plt.rcParams['axes.labelsize'] = 25
plt.rcParams['xtick.labelsize'] = 20  
plt.rcParams['ytick.labelsize'] = 20

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

categories = [str(idx) for idx in indices]
box_width = 0.28
margin = 0.34
box1 = ax1.boxplot(grad_ht_scale_all, 
                  positions=indices - margin/2, 
                  widths=box_width,
                  patch_artist=True,
                  boxprops=dict(facecolor='#1f77b4', alpha=0.7),
                  medianprops=dict(color='#404040', linewidth=1, linestyle='-'),
                  labels=categories,
                  showfliers=False)  
ax1.scatter(indices - margin/2, [max(x) for x in grad_ht_scale_all], 
           color='#1f77b4', marker='o', s=50, zorder=3)
box2 = ax1.boxplot(vol_gamma_all, 
                  positions=indices + margin/2, 
                  widths=box_width,
                  patch_artist=True,
                  boxprops=dict(facecolor='#ff7f0e', alpha=0.7),
                  medianprops=dict(color='#404040', linewidth=1, linestyle='-'),
                  labels=categories,
                  showfliers=False)
ax1.scatter(indices + margin/2, [max(x) for x in vol_gamma_all],
           color='#ff7f0e', marker='o', s=50, zorder=3)

ax1.set_xticks(indices)
ax1.set_xticklabels(categories)
ax1.tick_params(axis='y', labelsize=20)
ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax1.yaxis.offsetText.set_fontsize(22) 
ax1.grid(linestyle='-.', which='major')
plt.xticks(indices, indices)
ax1.legend([box1["boxes"][0], box2["boxes"][0]], 
          [r'$$C_{h_t}^0 / \epsilon_t$$', 
           r'$$\overline{\operatorname{vol}}(\mathcal{S}_t) / \overline{\gamma_t}$$'],
          loc='upper left',
          fontsize=18)
# legend_handles_ax1 = [box1["boxes"][0], box2["boxes"][0]]
# legend_labels_ax1 = [
#     r'$$C_{h_t}^0 / \epsilon_t}', 
#     r'$$\overline{\operatorname{vol}}(\mathcal{S}_t) / \overline{\gamma_t}}'
# ]

print("="*40)
alphas = [0.1, 0.01]
R = {}
for alpha in alphas:
    R[alpha] = []
    for idx_g, dataset in enumerate(datasets):
        dataset_path = os.path.join(path, dataset)
        if os.path.isdir(dataset_path):
            results_file = os.path.join(dataset_path, 'results')
            if os.path.exists(results_file):
                try:
                    filename_prefix = f"{method}_node"
                    filename_suffix = f"_alpha_{alpha}_eps_{eps}.npz"
                    
                    npz_files = [f for f in os.listdir(results_file) 
                                if f.startswith(filename_prefix) and f.endswith(filename_suffix)]
                    
                    if npz_files:
                        temp = []
                        for file in npz_files:
                            print(results_file+file)
                            re_ = np.load(os.path.join(results_file, file), allow_pickle=True)
                            grad_norms = [ii[0] for ii in re_['grad_norms']]
                            temp.append(np.max([val / grad_norms[0] for val in grad_norms]))
                        R[alpha].append((idx_g+1, np.mean(temp)))
                        # file = npz_files[0]
                        # print(results_file + file)
                        # for file in npz_files:  
                        # re_ = np.load(os.path.join(results_file, file), allow_pickle=True)
                        # grad_norms = [ii[0] for ii in re_['grad_norms']]
                        # R[alpha].append((idx_g+1, np.max([val / grad_norms[0] for val in grad_norms])))
                    else:
                        print(f"No matching .npz file found for {dataset} with method={method}, alpha={alpha}, eps={eps} in {results_file}")
                except Exception as e:
                        print(f"Erro  r loading results for {dataset}: {str(e)}")
       
plot_colors = ['#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
plot_markers = ['o', 's', '^', 'D', 'v', '*', 'p', 'X'] 
legend_handles_ax2 = []
legend_labels_ax2 = []

for i, alpha in enumerate(alphas): 
    x_data = [pair[0] for pair in R[alpha]]
    y_data = [pair[1] for pair in R[alpha]]
    scatter_plot = ax2.scatter(x_data, y_data, marker=plot_markers[i], s=50, color=plot_colors[i], label=rf'$\alpha={alpha}$')
    # legend_handles_ax2.append(scatter_plot)
    # legend_labels_ax2.append(rf'$\alpha={alpha}$')

ax2.tick_params(axis='y', labelsize=20)
ax2.tick_params(axis='x', labelsize=20)
ax2.set_ylabel(r"$R$", fontsize=25)
ax2.xaxis.set_ticks_position('top')
ax2.grid(linestyle='-.', which='major')
ax2.legend([r'$\alpha=10^{-1}$', r'$\alpha=10^{-2}$', r'$\alpha=10^{-3}$'],fontsize=18, loc='upper left') 
plt.subplots_adjust(hspace=0.001) 
plt.tight_layout()
plt.savefig('./figures/vol_gamma_boxplot.pdf', dpi=400, bbox_inches = 'tight')

