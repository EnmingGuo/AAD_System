# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import networkx as nx
"""
Created on 17-12-17

@summary: KShell算法节点重要性度量

@author: dreamhome
"""
def kshell(graph):
    """
    根据Kshell算法计算节点的重要性
    :param graph: 图
    :return: importance_dict{ks:[nodes]}
    """
    importance_dict = {}
    ks = 1
    while graph.nodes():
        # 暂存度为ks的顶点
        temp = []
        node_degrees_dict = graph.degree()
        # 每次删除度值最小的节点而不能删除度为ks的节点否则产生死循环。这也是这个算法存在的问题。
        kks = min(node_degrees_dict.values())
        while True:
            for k, v in node_degrees_dict.items():
                if v == kks:
                    temp.append(k)
                    graph.remove_node(k)
            node_degrees_dict = graph.degree()
            if kks not in node_degrees_dict.values():
                break
        importance_dict[ks] = temp
        ks += 1
    return importance_dict


import networkx as nx
import pandas as pd


def get_k_shell_vector(graph):
    k_shell_values = {}
    k_shell_vector = []
    max_k_shell = 0

    k_shell_dict = k_shell(graph)

    for level, nodes in k_shell_dict.items():
        for node in nodes:
            k_shell_values[node] = level
            max_k_shell = max(max_k_shell, level)

    for i in range(90):
        if i in k_shell_values:
            k_shell_vector.append(k_shell_values[i])
        else:
            k_shell_vector.append(float('inf'))

    return k_shell_vector, max_k_shell

def k_shell(graph):
    importance_dict = {}
    level = 1
    while len(graph.degree):
        importance_dict[level] = []
        while True:
            level_node_list = []
            for item in graph.degree:
                if item[1] <= level:
                    level_node_list.append(item[0])
            graph.remove_nodes_from(level_node_list)
            importance_dict[level].extend(level_node_list)
            if not len(graph.degree):
                return importance_dict
            if min(graph.degree, key=lambda x: x[1])[1] > level:
                break
        level = min(graph.degree, key=lambda x: x[1])[1]
    return importance_dict



if __name__ == "__main__":
    graph = nx.Graph()
    graph.add_edges_from(
        [(1, 4), (2, 4), (3, 4), (4, 5), (5, 6), (5, 7), (3, 5), (6, 7)])
    print (k_shell(graph))


