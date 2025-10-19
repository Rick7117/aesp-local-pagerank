import random, os
from utils import graph, read_files
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pandas as pd

# python main.py --graph com-dblp --method xinit yinit zeroinit --node 20 --alpha 0.01 --eps 1e-7
path_y = './datasets/com-dblp/results/yinit_node_20_alpha_0.01_eps_1e-07.npz'
path_x = './datasets/com-dblp/results/xinit_node_20_alpha_0.01_eps_1e-07.npz'
path_0 = './datasets/com-dblp/results/zeroinit_node_20_alpha_0.01_eps_1e-07.npz'

re_ = np.load(path_x, allow_pickle=True)
err_x, oper_x, runtimeacc_x = re_['errs'], re_['opers'], re_['runtime_acc']
re_ = np.load(path_y, allow_pickle=True)
err_y, oper_y, runtimeacc_y = re_['errs'], re_['opers'], re_['runtime_acc']
re_ = np.load(path_0, allow_pickle=True)
err_0, oper_0, runtimeacc_0 = re_['errs'], re_['opers'], re_['runtime_acc']

sns.set_theme(style="white")
warnings.filterwarnings("ignore")
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
plt.rcParams['xtick.labelsize'] = 20  
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['axes.labelsize'] = 25

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].plot(np.cumsum(oper_x), np.log(err_x), label=r'$\boldsymbol{z}_{t}^{(0)} = \boldsymbol{x}^{(t-1)}$', linewidth=2, ls='--')  
ax[0].plot(np.cumsum(oper_y), np.log(err_y), label=r'$\boldsymbol{z}_{t}^{(0)} = \boldsymbol{y}^{(t-1)}$', linewidth=2, ls='--')  
ax[0].plot(np.cumsum(oper_0), np.log(err_0), label=r'$\boldsymbol{z}_{t}^{(0)} = \boldsymbol{0}$', linewidth=2, ls='--') 

ax[0].grid(linestyle='-.', which='major')
ax[0].set_xlabel('\# of Operations', fontsize=20)
ax[0].set_ylabel(r'$\log \|\boldsymbol{D}^{-1}(\hat\pi-\pi)\|_{\infty}$', fontsize=20)

ax[1].grid(linestyle='-.', which='major')
ax[1].plot(runtimeacc_x, np.log(err_x), label=r'$\boldsymbol{z}_{t}^{(0)} = \boldsymbol{x}^{(t-1)}$', linewidth=2, ls='--')  
ax[1].plot(runtimeacc_y, np.log(err_y), label=r'$\boldsymbol{z}_{t}^{(0)} = \boldsymbol{y}^{(t-1)}$', linewidth=2, ls='--') 
ax[1].plot(runtimeacc_0, np.log(err_0), label=r'$\boldsymbol{z}_{t}^{(0)} = \boldsymbol{0}$', linewidth=2, ls='--') 
ax[1].set_xlabel('Running Times (s)', fontsize=20)
ax[1].legend(fontsize=20)
plt.tight_layout()
plt.savefig("figures/init_start.pdf", format="pdf", bbox_inches="tight")

