import numpy as np
import time
import scipy.sparse as sp
import os, json, pickle
from ppr_solver import appr, apprOpt, locGD, aespAPPR, aespLocGD
from ppr_solver import ista, fista, aspr, cheby
from ppr_solver import aespAPPR_init

def read_files(path, graph_list=None):
    graph_paths = {}
    
    for graph_name in os.listdir(path):
        graph_dir = os.path.join(path, graph_name)
        if os.path.isdir(graph_dir) and (graph_list is None or graph_name in graph_list):
            for filename in os.listdir(graph_dir):
                if filename.endswith('-mat.npz'):
                    data_file = os.path.join(graph_dir, filename)
                    graph_paths[graph_name] = data_file
    return graph_paths

class graph:
    def __init__(self, name, path):
        self.adj_m = sp.load_npz(path)
        self.adj_m = self.adj_m.tocsr()
        if self.adj_m.shape[0] != self.adj_m.shape[1]:
            raise ValueError(f"Adjacency matrix must be square, current shape is {self.adj_m.shape}")
        self.n = self.adj_m.shape[0]
        self.m = self.adj_m.nnz // 2 
        if type(self.adj_m) == sp.csr_matrix:
            self.degree = self.adj_m.sum(1).A1.astype(np.int64)
        else:
            self.degree = self.adj_m.sum(1).astype(np.int64)
        self.indices = self.adj_m.indices
        self.indptr = self.adj_m.indptr
        self.name = name
        self.dataset_path = os.path.dirname(path)

        print("="*40 + "\nGraph Statistics\n" + "="*40)
        print(f"Graph Path: {self.dataset_path}")
        print(f"Graph {name} loaded. n: {self.n}, m: {self.m}")

    def opt_solution(self, node, alpha):
        n, indptr, indices, degree = self.n, self.indptr, self.indices, self.degree
        s = np.zeros(n)
        s[node] = 1.

        opt_eps = 1e-5/n
        re_ = apprOpt(n, indptr, indices, degree, s, alpha, opt_eps, opt_x=None)
        print(f'(time = {re_[3]:.4f}s) finish {self.name} (node={node}, alpha={alpha:.1e}, eps={opt_eps:.1e}): l1_norm_opt_x: {np.linalg.norm(re_[0], 1):.5f}')
        self.opt_x = re_[0]

        save_dir = os.path.join(self.dataset_path, 'opt_solutions')
        os.makedirs(save_dir, exist_ok=True)
        filename = f'node_{node}_alpha_{alpha}.npz'
        save_path = os.path.join(save_dir, filename)
        np.savez(save_path, solution=re_[0])
        return self.opt_x

    def load_opt_solution(self, node, alpha):
        save_dir = os.path.join(self.dataset_path, 'opt_solutions')
        filename = f'node_{node}_alpha_{alpha}.npz'
        save_path = os.path.join(save_dir, filename)
        
        try:
            loaded = np.load(save_path, allow_pickle=True)
            opt_x = loaded['solution']
            print(f'load {self.name}\'s opt solution (node = {node}, alpha={alpha:.1e}): {np.linalg.norm(opt_x, 1):.5f}')
            return opt_x
        except FileNotFoundError:
            print(f"Solution file not found for node {node} with alpha value {alpha}")
            return None
            
    def run_method(self, algo, node, alpha, eps):
        opt_x = self.load_opt_solution(node, alpha)
        if opt_x is None:
            opt_x = self.opt_solution(node, alpha)
        s = np.zeros(self.n)
        s[node] = 1.
        if algo == "appr":
            result = appr(self.n, self.indptr, self.indices, self.degree, s, alpha, eps, opt_x=opt_x)
        elif algo == 'appropt':
            result = apprOpt(self.n, self.indptr, self.indices, self.degree, s, alpha, eps, opt_x=opt_x)
        elif algo == 'locgd':
            result = locGD(self.n, self.indptr, self.indices, self.degree, s, alpha, eps, opt_x=opt_x)
        elif algo == "aespappr":
            result = aespAPPR(self.n, self.indptr, self.indices, self.degree, s, alpha, eps, opt_x=opt_x)
        elif algo == "aesplocgd":
            result = aespLocGD(self.n, self.indptr, self.indices, self.degree, s, alpha, eps, opt_x=opt_x)
        elif algo == 'ista':
            ista_eps = 7
            result = ista(self.n, self.indptr, self.indices, self.degree, s, alpha, ista_eps, eps/(1+ista_eps), opt_x, l1_err=None)
        elif algo == 'fista':
            fista_eps = 1e-2
            result = fista(self.n, self.indptr, self.indices, self.degree, s, alpha, fista_eps, eps, False, opt_x, l1_err=None)
        elif algo == 'cheby':
            result = cheby(self.n, self.indptr, self.indices, self.degree, s, alpha, eps, opt_x)
        elif algo == 'aspr':
            aspr_eps = 0.1
            result = aspr(self.n, self.indptr, self.indices, self.degree, s, alpha, aspr_eps, eps/(1+aspr_eps), opt_x)
        elif algo == 'xinit':
            result = aespAPPR_init(self.n, self.indptr, self.indices, self.degree, s, alpha, eps, opt_x=opt_x, init='x')
        elif algo == 'yinit':
            result = aespAPPR_init(self.n, self.indptr, self.indices, self.degree, s, alpha, eps, opt_x=opt_x, init='y')
        elif algo == 'zeroinit':
            result = aespAPPR_init(self.n, self.indptr, self.indices, self.degree, s, alpha, eps, opt_x=opt_x, init='zero')
        
        print(f'finish {self.name} with {algo} (node={node}, alpha={alpha:.1e}, eps={eps:.1e}): l1_norm_est_x: {np.linalg.norm(result[0], 1):.5f}')
        return result
