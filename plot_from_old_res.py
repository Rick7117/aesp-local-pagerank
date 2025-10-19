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

fig, ax = plt.subplots(1, 1, figsize=(5, 4))

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

path = '/mnt/data/binbin/git/ICML_2025_code_review/results/ogb-mag240m'
files = os.listdir(path)
files.sort()
color1 = [
    '#D62728',
    '#FF7F0E'
] 
color2 = [
    '#588157',
    '#1F77B4',
    '#6D597A',
    '#8C8C8C',
]
n_col1, n_col2 = 0, 0
for i, file in enumerate(files):
    if i == 5: break
    file_path = os.path.join(path, file)
    data = np.load(file_path, allow_pickle=True)
    opers, errs = data['opers'], data['errs']
    if file.startswith('AESP'):
        color = color1[n_col1]
        n_col1 += 1
    else:
        color = color2[n_col2]
        n_col2 += 1
    ax.plot(np.cumsum(opers), np.log(errs), linewidth=1.5, ls='--', color=color) 

ax.legend(['AESP-LocAPPR', 'AESP-LocGD', 'APPR-Opt', 'APPR', 'LocGD'], fontsize=12)
ax.grid(linestyle='-.', which='major')
ax.set_ylabel(r'$\log \|\boldsymbol{D}^{-1}(\hat\pi-\pi)\|_{\infty}$')
ax.set_xlabel('\# of Operations')
ax.set_title('ogb-mag240m', fontsize=22)
plt.tight_layout()
plt.savefig("figures/convergence_from_old_one_plot.pdf", format="pdf", bbox_inches="tight")







