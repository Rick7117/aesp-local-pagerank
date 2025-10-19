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
plt.rcParams['xtick.labelsize'] = 18  
plt.rcParams['ytick.labelsize'] = 18 
plt.rcParams['axes.labelsize'] = 25

path = './datasets/com-dblp/results/aespappr_node_0_alpha_0.01_eps_1e-08.npz'
results_data = np.load(path, allow_pickle=True)
grad_norms = [ii[0] for ii in results_data['grad_norms']]
eps_t = results_data['eps_t']
grad_norms_scale = [grad_norms[i] / eps_t[i] for i in range(len(grad_norms))]
vols_bar = [np.mean(vol_st) for vol_st in results_data['vols']] 
gammas_bar = [np.mean(gamma_t) for gamma_t in results_data['gammas']]
vol_gamma = [vols_bar[i] / gammas_bar[i] for i in range(len(vols_bar))]


plt.plot(vol_gamma)
plt.plot(grad_norms_scale)

plt.savefig('./figures/vol_gamma_upperbound.pdf')