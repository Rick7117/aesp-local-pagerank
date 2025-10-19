# import random, os
# from utils import graph, read_files
# import numpy as np

# import matplotlib
# import matplotlib.pyplot as plt
# from matplotlib.ticker import ScalarFormatter
# import seaborn as sns
# import warnings
# import pandas as pd

# sns.set_theme(style="white")
# warnings.filterwarnings("ignore")
# plt.rc('text', usetex=True)
# plt.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
# plt.rcParams['xtick.labelsize'] = 16  
# plt.rcParams['ytick.labelsize'] = 16
# plt.rcParams['axes.labelsize'] = 20


# fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# path = '/mnt/data/binbin/git/ICML_2025_code_review/results/com-dblp'
# files = os.listdir(path)
# files.sort()

# for i, file in enumerate(files):
#     if i == 5: break
#     file_path = os.path.join(path, file)
#     data = np.load(file_path, allow_pickle=True)
#     opers, errs = data['opers'], data['errs']
#     ax[0].plot(np.cumsum(opers), np.log(errs), linewidth=1.5, ls='--') 

# ax[0].grid(linestyle='-.', which='major')
# ax[0].set_xlabel('\# of Operations')
# ax[0].set_ylabel(r'$\log \|\boldsymbol{D}^{-1}(\hat\pi-\pi)\|_{\infty}$')
# ax[0].set_title('com-dblp', fontsize=25)

# path = '/mnt/data/binbin/git/ICML_2025_code_review/results/ogb-mag240m'
# files = os.listdir(path)
# files.sort()

# for i, file in enumerate(files):
#     if i == 5: break
#     file_path = os.path.join(path, file)
#     data = np.load(file_path, allow_pickle=True)
#     opers, errs = data['opers'], data['errs']
#     ax[1].plot(np.cumsum(opers), np.log(errs), linewidth=1.5, ls='--') 

# ax[1].legend(['AESP-LocAPPR', 'AESP-LocGD', 'APPR Opt', 'APPR', 'LocGD'], fontsize=13)
# ax[1].grid(linestyle='-.', which='major')
# ax[1].set_xlabel('\# of Operations')
# ax[1].set_title('ogb-mag240m', fontsize=25)
# plt.tight_layout()
# plt.savefig("figures/convergence_from_old.pdf", format="pdf", bbox_inches="tight")

import random, os
from utils import graph, read_files
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import warnings
import pandas as pd

sns.set_theme(style="white")
warnings.filterwarnings("ignore")
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
plt.rcParams['xtick.labelsize'] = 12  
plt.rcParams['ytick.labelsize'] = 12  
plt.rcParams['axes.labelsize'] = 18
# plt.rcParams['axes.titleweight'] = 'bold'

# Create 5 subplots for different eps values
fig, axes = plt.subplots(1, 5, figsize=(20, 4))

# Define methods and eps values
methods = ['aespappr', 'aesplocgd', 'locgd', 'appr', 'appropt']
eps_values = ['1e-06', '1e-07', '1e-08', '1e-09', '1e-10']
eps_labels = ['1e-6', '1e-7', '1e-8', '1e-9', '1e-10']

# Define colors for each method
colors = {
    'aespappr': '#D62728',
    'aesplocgd': '#FF7F0E', 
    'locgd': '#588157',
    'appr': '#1F77B4',
    'appropt': '#6D597A'
}

# Define method labels for legend
method_labels = {
    'aespappr': 'AESP-LocAPPR',
    'aesplocgd': 'AESP-LocGD',
    'locgd': 'LocGD',
    'appr': 'APPR',
    'appropt': 'APPR-Opt'
}

path = '/mnt/data/binbin/git/NIPS2025/datasets/ogbn-arxiv/results'

# Initialize list to collect data for CSV export
csv_data = []

# Plot each eps value in a separate subplot
for i, eps in enumerate(eps_values):
    ax = axes[i]
    
    # Plot each method for current eps
    for method in methods:
        filename = f"{method}_node_117_alpha_0.1_eps_{eps}.npz"
        file_path = os.path.join(path, filename)
        
        if os.path.exists(file_path):
            data = np.load(file_path, allow_pickle=True)
            opers, errs = data['opers'], data['errs']
            cumsum_opers = np.cumsum(opers)
            log_errs = np.log(errs)
            
            # Collect data for CSV export
            for j in range(len(cumsum_opers)):
                csv_data.append({
                    'method': method,
                    'eps': eps_labels[i],
                    'cumsum_opers': cumsum_opers[j],
                    'log_errs': log_errs[j]
                })
            
            ax.plot(cumsum_opers, log_errs, 
                   linewidth=1.5, ls='--', color=colors[method], 
                   label=method_labels[method])
    
    # Configure subplot
    ax.set_ylabel(r'$\log \|\boldsymbol{D}^{-1}(\hat\pi-\pi)\|_{\infty}$')
    ax.set_xlabel('\# of Operations')
    if i == 0:  # Only set ylabel for leftmost subplot
        ax.set_ylabel(r'$\log \|\boldsymbol{D}^{-1}(\hat\pi-\pi)\|_{\infty}$')
    ax.grid(linestyle='-.', which='major')
    ax.set_title(f'eps = {eps_labels[i]}', fontsize=16)
    
    # Add legend only to the last subplot
    if i == len(eps_values) - 1:
        ax.legend(fontsize=10, loc='upper right')

plt.tight_layout()
plt.savefig("figures/convergence_eps_comparison_ogbn_arxiv.pdf", format="pdf", bbox_inches="tight")
plt.savefig("figures/convergence_eps_comparison_ogbn_arxiv.png", format="png", bbox_inches="tight", dpi=300)

# Export data to CSV
import pandas as pd
df = pd.DataFrame(csv_data)

# Create tables directory if it doesn't exist
tables_dir = '/mnt/data/binbin/git/NIPS2025/tables'
os.makedirs(tables_dir, exist_ok=True)

# Save to CSV file
csv_filename = os.path.join(tables_dir, 'convergence_eps_comparison_ogbn_arxiv.csv')
df.to_csv(csv_filename, index=False)
print(f"Data exported to {csv_filename}")
print(f"CSV contains {len(df)} rows with columns: {list(df.columns)}")
print("First 5 rows:")
print(df.head())







