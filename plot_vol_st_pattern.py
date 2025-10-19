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
plt.rcParams['axes.labelsize'] = 25

path = './datasets/com-dblp/com-dblp_csr-mat.npz'
g = graph('com-dblp', path)

# re_ = g.run_method('ista', 1, 0.02, 1e-7)
# vol_ista = re_[2]
# len_ista = len(vol_ista)

re_ = g.run_method('appr', 1, 0.03, 1e-7)
vol_appr = re_[2]
print(max(vol_appr))
len(vol_appr)

re_ = g.run_method('aespappr', 1, 0.03, 1e-7)
vol_aespappr = re_[2]
print(max(vol_aespappr))

# x_ista = np.linspace(0, 1, len(vol_ista))
# x_appr = np.linspace(0, 1, len(vol_appr))
# x_aespappr = np.linspace(0, 1.5, len(vol_aespappr))

# plt.plot(x_ista, vol_ista, label='ISTA', color='blue')
# plt.plot(x_appr, vol_appr, label=r'APPR', color='green')
# plt.plot(x_aespappr, vol_aespappr, label=r'AESP-PPR', color='red')
# plt.plot(vol_ista, label='ISTA', color='blue')
cut = 87
vol_appr = vol_appr[:cut]
plt.plot(vol_appr, label=r'APPR', color='blue')
plt.plot(vol_aespappr, label=r'Ours', color='red')
# plt.axhline(y=np.sum(g.degree), color='grey', linestyle='--')

# plt.fill_between(range(len(vol_ista)), vol_ista, color='blue', alpha=0.2)
plt.fill_between(range(len(vol_appr)), vol_appr, color='blue', alpha=0.2)
plt.fill_between(range(len(vol_aespappr)), vol_aespappr, color='red', alpha=0.2)

plt.ylim(0, max(max(vol_appr), max(vol_aespappr)) * 1.3)
plt.legend(
    fontsize=18,
    loc = 'upper left'
)
plt.xticks([]) 
plt.yticks([]) 

plt.xlabel(r'Iterations ($t$)')
# plt.ylabel(r'$\operatorname{vol}(\mathcal{S}_t) \text{ or } \operatorname{vol}(\mathcal{S}_t^{(k+1)}) $')
plt.ylabel(r'\# of Operations')
plt.tight_layout()
plt.savefig('./figures/vol_st_pattern.pdf', dpi=400, bbox_inches = 'tight')


