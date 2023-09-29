import numpy as np
import pandas as pd
import networkx as nx
from K_Shell import k_shell;
from K_Shell import kshell
import os
import scipy.io as sio
edges = pd.DataFrame()

edges['sources'] = [0,1,1,2,2,2,3,3,4,4,5,5,5]

edges['targets'] = [2,4,5,3,0,1,2,5,1,5,1,3,4]

edges['weights'] = [1,1,1,1,1,1,1,1,1,1,1,1,1]

G = nx.from_pandas_edgelist(edges,source='sources',target='targets',edge_attr='weights')

def get_graph():
    filePath = 'C:\\Users\lijl7\Desktop\AAD_System\Mat'
    for root, dirs, files in os.walk(filePath):

        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list

        # 遍历文件
        for f in files:
            Path = os.path.join(root, f)
            load_data = sio.loadmat(Path)
            matrix = load_data['features']

            a = get_degree(matrix)
            b = get_degree_centrality(matrix)
            c = get_betweeness_centrality(matrix)
            g = get_pagerank(matrix)
            d = get_degree_eigenvector(matrix)
            e = get_closeness_centrality(matrix)
            h = get_flow_efficencies(matrix)
            l = get_KS(matrix)[0]
            m = get_local_efficiency(matrix)

    return G

def local_efficiency(graph):
    local_efficiency_values = {}

    for node in graph.nodes:
        neighbors = set(graph.neighbors(node))
        neighbors.add(node)  # Include the node itself in the neighbors set
        subgraph = graph.subgraph(neighbors)
        local_efficiency_values[node] = nx.global_efficiency(subgraph)

    return local_efficiency_values
def get_local_efficiency(matrix):
    edges = pd.DataFrame()
    sources = []
    targets = []
    weights = []

    for row in range(0, 90):
        for col in range(0, 90):
            if matrix[row][col] == 1:
                sources.append(row)
                targets.append(col)
                weights.append(1)

    edges['sources'] = sources
    edges['targets'] = targets
    edges['weights'] = weights

    # 指定节点数量
    num_nodes = 90

    # 创建一个无向图
    G = nx.Graph()

    # 添加节点
    G.add_nodes_from(range(0, num_nodes))

    # 添加边
    edges_list = [(sources[i], targets[i], {'weight': weights[i]}) for i in range(len(sources))]
    G.add_edges_from(edges_list)
    local_eff = local_efficiency(G)
    res = []
    for i in range(0, 90):
        res.append(local_eff[i])

    return res

def get_degree(matrix):
    edges = pd.DataFrame()
    sources = []
    targets = []
    weights = []

    for row in range(0, 90):
        for col in range(0, 90):
            if matrix[row][col] == 1:
                sources.append(row)
                targets.append(col)
                weights.append(1)

    edges['sources'] = sources
    edges['targets'] = targets
    edges['weights'] = weights

    # 指定节点数量
    num_nodes = 90

    # 创建一个无向图
    G = nx.Graph()

    # 添加节点
    G.add_nodes_from(range(0, num_nodes))

    # 添加边
    edges_list = [(sources[i], targets[i], {'weight': weights[i]}) for i in range(len(sources))]
    G.add_edges_from(edges_list)

    # 计算每个节点的度数
    degree = dict(nx.degree(G))
    res = []
    for i in range(0,90):
        res.append(degree[i])

    return res
def get_flow_efficencies(matrix):
    edges = pd.DataFrame()
    sources = []
    targets = []
    weights = []

    for row in range(0, 90):
        for col in range(0, 90):
            if matrix[row][col] == 1:
                sources.append(row)
                targets.append(col)
                weights.append(1)

    edges['sources'] = sources
    edges['targets'] = targets
    edges['weights'] = weights

    # 指定节点数量
    num_nodes = 90

    # 创建一个无向图
    G = nx.Graph()

    # 添加节点
    G.add_nodes_from(range(0, num_nodes))

    # 添加边
    edges_list = [(sources[i], targets[i], {'weight': weights[i]}) for i in range(len(sources))]
    G.add_edges_from(edges_list)

    # 获取全部的连通分量
    connected_subgraphs = nx.connected_components(G)

    # 字典来存储节点的流量系数
    flow_coefficients = {}

    # 对每个连通子图计算流量系数并存储到字典中
    for subgraph_nodes in connected_subgraphs:
        subgraph = G.subgraph(subgraph_nodes)
        subgraph_flow_coefficients = nx.current_flow_closeness_centrality(subgraph)
        flow_coefficients.update(subgraph_flow_coefficients)
    # 计算每个节点的度数

    res = []
    for i in range(0, 90):
        res.append(flow_coefficients[i])
    res = [0 if x == float('inf') else x for x in res]
    return res
def get_degree_centrality(matrix):
    edges = pd.DataFrame()
    sources = []
    targets = []
    weights = []

    for row in range(0, 90):
        for col in range(0, 90):
            if matrix[row][col] == 1:
                sources.append(row)
                targets.append(col)
                weights.append(1)

    edges['sources'] = sources
    edges['targets'] = targets
    edges['weights'] = weights

    # 指定节点数量
    num_nodes = 90

    # 创建一个无向图
    G = nx.Graph()

    # 添加节点
    G.add_nodes_from(range(0, num_nodes))

    # 添加边
    edges_list = [(sources[i], targets[i], {'weight': weights[i]}) for i in range(len(sources))]
    G.add_edges_from(edges_list)

    # 计算每个节点的度数
    degree_centrality = dict(nx.degree_centrality(G))
    res = []
    for i in range(0, 90):
        res.append(degree_centrality[i])
    res = res
    return res
def get_degree_eigenvector(matrix):
    edges = pd.DataFrame()
    sources = []
    targets = []
    weights = []

    for row in range(0, 90):
        for col in range(0, 90):
            if matrix[row][col] == 1:
                sources.append(row)
                targets.append(col)
                weights.append(1)

    edges['sources'] = sources
    edges['targets'] = targets
    edges['weights'] = weights

    # 指定节点数量
    num_nodes = 90

    # 创建一个无向图
    G = nx.Graph()

    # 添加节点
    G.add_nodes_from(range(0, num_nodes))

    # 添加边
    edges_list = [(sources[i], targets[i], {'weight': weights[i]}) for i in range(len(sources))]
    G.add_edges_from(edges_list)

    # 计算每个节点的度数
    eigenvector_centrality = dict(nx.eigenvector_centrality(G))
    res = []
    for i in range(0, 90):
        res.append(eigenvector_centrality[i])
    res = res
    return res
def get_betweeness_centrality(matrix):
    edges = pd.DataFrame()
    sources = []
    targets = []
    weights = []

    for row in range(0, 90):
        for col in range(0, 90):
            if matrix[row][col] == 1:
                sources.append(row)
                targets.append(col)
                weights.append(1)

    edges['sources'] = sources
    edges['targets'] = targets
    edges['weights'] = weights

    # 指定节点数量
    num_nodes = 90

    # 创建一个无向图
    G = nx.Graph()

    # 添加节点
    G.add_nodes_from(range(0, num_nodes))

    # 添加边
    edges_list = [(sources[i], targets[i], {'weight': weights[i]}) for i in range(len(sources))]
    G.add_edges_from(edges_list)

    # 计算每个节点的度数
    betweeness_centrality = dict(nx.betweenness_centrality(G))
    res = []
    for i in range(0, 90):
        res.append(betweeness_centrality[i])
    res = res
    return res
def get_closeness_centrality(matrix):
    edges = pd.DataFrame()
    sources = []
    targets = []
    weights = []

    for row in range(0, 90):
        for col in range(0, 90):
            if matrix[row][col] == 1:
                sources.append(row)
                targets.append(col)
                weights.append(1)

    edges['sources'] = sources
    edges['targets'] = targets
    edges['weights'] = weights

    # 指定节点数量
    num_nodes = 90

    # 创建一个无向图
    G = nx.Graph()

    # 添加节点
    G.add_nodes_from(range(0, num_nodes))

    # 添加边
    edges_list = [(sources[i], targets[i], {'weight': weights[i]}) for i in range(len(sources))]
    G.add_edges_from(edges_list)

    # 计算每个节点的度数
    closeness_centrality = dict(nx.closeness_centrality(G))
    res = []
    for i in range(0, 90):
        res.append(closeness_centrality[i])
    res = res
    return res
def get_pagerank(matrix):
    edges = pd.DataFrame()
    sources = []
    targets = []
    weights = []

    for row in range(0, 90):
        for col in range(0, 90):
            if matrix[row][col] == 1:
                sources.append(row)
                targets.append(col)
                weights.append(1)

    edges['sources'] = sources
    edges['targets'] = targets
    edges['weights'] = weights

    # 指定节点数量
    num_nodes = 90

    # 创建一个无向图
    G = nx.Graph()

    # 添加节点
    G.add_nodes_from(range(0, num_nodes))

    # 添加边
    edges_list = [(sources[i], targets[i], {'weight': weights[i]}) for i in range(len(sources))]
    G.add_edges_from(edges_list)

    # 计算每个节点的度数
    pagerank = dict(nx.pagerank(G))
    res = []
    for i in range(0, 90):
        res.append(pagerank[i])
    res = res
    return res
from K_Shell import get_k_shell_vector
def get_KS(matrix):
    edges = pd.DataFrame()
    sources = []
    targets = []
    weights = []

    for row in range(0, 90):
        for col in range(0, 90):
            if matrix[row][col] == 1:
                sources.append(row)
                targets.append(col)
                weights.append(1)

    edges['sources'] = sources
    edges['targets'] = targets
    edges['weights'] = weights

    # 指定节点数量
    num_nodes = 90

    # 创建一个无向图
    G = nx.Graph()

    # 添加节点
    G.add_nodes_from(range(0, num_nodes))

    # 添加边
    edges_list = [(sources[i], targets[i], {'weight': weights[i]}) for i in range(len(sources))]
    G.add_edges_from(edges_list)

    # 计算每个节点的度数
    ks_values = get_k_shell_vector(G)
    return ks_values

# 测试部分
#get_graph()

# #degree 选择
# degree_list = nx.degree(G)
# print(nx.degree(G))
# print(degree_list[1])
# #connected compnents
# print(nx.connected_components(G))
# #度中心性 选择
# print(nx.degree_centrality(G))
# #特征向量中心性 选择
# print(nx.eigenvector_centrality(G))
# #betweenness 选择
# print(nx.betweenness_centrality(G))
# #closeness 选择
# print(nx.closeness_centrality(G))
# #pagerank 选择
# print(nx.pagerank(G))
