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
plt.rcParams['xtick.labelsize'] = 12  
plt.rcParams['ytick.labelsize'] = 12  
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 25

fig, ax = plt.subplots(1, 1, figsize=(5, 4))

path = './datasets/com-dblp/results'
endStr = 'node_20_alpha_0.01_eps_1e-08.npz'
methods = ['aespappr', 'aesplocgd', 'cheby', 'fista']
matching_files = [f for f in os.listdir(path) 
                 if f.endswith(endStr) and any(f.startswith(method) for method in methods)]
matching_files.sort()
print(matching_files)

for filename in matching_files:
    filepath = os.path.join(path, filename)
    re_ = np.load(filepath, allow_pickle=True)
    opers = re_['opers']
    errs = re_['errs']
    if filename.startswith('fista'):
        cut = 75
        ax.plot(np.cumsum(opers)[:cut], np.log(errs)[:cut], linewidth=1.5, ls='--') 
    else:
        ax.plot(np.cumsum(opers), np.log(errs), linewidth=1.5, ls='--')  

ax.grid(linestyle='-.', which='major')
ax.set_xlabel('\# of Operations')
ax.set_ylabel(r'$\log \|\boldsymbol{D}^{-1}(\hat\pi-\pi)\|_{\infty}$')

# ax.legend(['LocCH', 'FISTA'], fontsize=13)
ax.legend(['AESP-LocAPPR', 'AESP-LocGD', 'LocCH', 'FISTA'], fontsize=13)
ax.grid(linestyle='-.', which='major')

plt.tight_layout()
plt.savefig("figures/fast_convergence.pdf", format="pdf", bbox_inches="tight")

