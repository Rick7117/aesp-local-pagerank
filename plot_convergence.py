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

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

path = './datasets/method/results'
endStr = 'node_20_alpha_0.01_eps_1e-07.npz'
matching_files = [f for f in os.listdir(path) 
                 if f.endswith(endStr)]
matching_files.sort()
print(matching_files)
for filename in matching_files:
    filepath = os.path.join(path, filename)
    re_ = np.load(filepath, allow_pickle=True)
    opers = re_['opers']
    errs = re_['errs']
    runtimeacc = re_['runtime_acc']
    ax[0].plot(np.cumsum(opers), np.log(errs), linewidth=1.5, ls='--')  
    ax[1].plot(runtimeacc, np.log(errs), linewidth=1.5, ls='--')

ax[0].grid(linestyle='-.', which='major')
ax[0].set_xlabel('\# of Operations')
ax[0].set_ylabel(r'$\log \|\boldsymbol{D}^{-1}(\hat\pi-\pi)\|_{\infty}$')

ax[1].legend(['AESP-LocAPPR', 'AESP-LocGD', 'APPR', 'APPR Opt', 'LocGD'], fontsize=13)
ax[1].grid(linestyle='-.', which='major')
ax[1].set_xlabel('Running Times (s)')

plt.tight_layout()
plt.savefig("figures/convergence_as-skitter.pdf", format="pdf", bbox_inches="tight")

