from utils import graph, read_files
import pandas as pd

path = './datasets'
graph_data = []
file_path_dict = read_files(path)
for graph_name, graph_path in file_path_dict.items():
    g = graph(graph_name, graph_path)
    graph_data.append({
        'name': graph_name,
        'n': g.n,
        'm': g.m,
    })
graph_data = pd.DataFrame(graph_data)
graph_data.to_csv('./tables/graph_statistic.csv', index=False)
graph_data.to_latex('./tables/graph_statistic.tex', index=False)

