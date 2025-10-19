import random, os, gc, sys
import argparse
import numpy as np
import multiprocessing
from utils import graph, read_files

np.random.seed(2025)

def process_graph(graph_name, graph_path, method, node, alpha, eps):
    g = graph(graph_name, graph_path)  

    if node == -1:
        node = random.choice(range(g.n))

    x, errs, opers, runtime, runtime_acc, grad_norms, vols, gammas, eps_t = g.run_method(method, node, alpha, eps)

    try:
        grad_norms_arr = np.array([list(grad_norm) for grad_norm in grad_norms], dtype=object)
        vols_arr = np.array([list(vol) for vol in vols], dtype=object)
        gammas_arr = np.array([list(gamma) for gamma in gammas], dtype=object)
    except:
        grad_norms_arr = np.array(grad_norms, dtype=object)
        vols_arr = np.array(vols, dtype=object)
        gammas_arr = np.array(gammas, dtype=object)
    result_data = {
        'x': x,
        'errs': errs,
        'opers': opers,
        'runtime': runtime,
        'runtime_acc': runtime_acc,
        'grad_norms': grad_norms_arr,
        'vols': vols_arr,
        'gammas': gammas_arr,
        'eps_t': eps_t
    }
    save_dir = os.path.join(g.dataset_path, 'results')
    os.makedirs(save_dir, exist_ok=True)
    
    filename = f'{method}_node_{node}_alpha_{alpha}_eps_{eps}.npz'
    save_path = os.path.join(save_dir, filename)
    np.savez(save_path, **result_data)
    print(f"Finish saving {graph_name}!")
    del g
    gc.collect()

def process_graph_wrapper(args):
    try:
        return process_graph(*args)
    except Exception as e:
        print(f"error: {str(e)}")
        raise 

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='./datasets', help='path to the graph dataset')
parser.add_argument('--method', type=str, nargs='+', default='AESP_locAPPR', help='algorithm name, can be a string or a list of strings')
parser.add_argument('--graph', type=str, nargs='+', default=None, help='graph names, can be a string or a list of strings')
parser.add_argument('--node', type=int, nargs='+', default=[-1], help='node indices, can be a single value or a list')
parser.add_argument('--nodenum', type=int, default=None, help='number of nodes to sample when node not specified')
parser.add_argument('--alpha', type=float, nargs='+', default=[0.1], help='alpha values, can be a single value or a list')
parser.add_argument('--eps', type=float, nargs='+', default=[1e-6], help='epsilon values, can be a single value or a list')

args = parser.parse_args()

methods, graphs = args.method, args.graph 
alphas, epses = args.alpha, args.eps
path = args.path

valid_methods = [
    'appr', 'appropt', 'locgd', 'aespappr', 'aesplocgd', 
    'cheby', 'ista', 'fista', 'aspr', 'cheby',
    'xinit', 'yinit', 'zeroinit'
    ]
if not all(method.lower() in valid_methods for method in methods):
    invalid_methods = [method for method in methods if method.lower() not in valid_methods]
    print(f"Error: Unsupported algorithm name(s): {invalid_methods}. Valid algorithms are: {valid_methods}")
    sys.exit(1)

random.seed(2025)
if args.nodenum is not None:
    # nodes = [-1] * args.nodenum
    nodes = random.sample(range(1000), args.nodenum)
else:
    nodes = args.node

num_cpus = multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=num_cpus)
print(f"Number of CPUs: {num_cpus}")

arg_lists = []
file_path_dict = read_files(path, graphs)
for graph_name, graph_path in file_path_dict.items():
    for node in nodes:
        for alpha in alphas:
            for method in methods:
                for eps in epses:
                    arg_lists.append((graph_name, graph_path, method, node, alpha, eps))

pool.map(process_graph_wrapper, arg_lists)
pool.close()
pool.join()
