import torch
import numpy as np
import seaborn as sb
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_series(x : np.array, columns : list):
    data = dict.fromkeys(columns)
    for i ,s in enumerate(columns):
        data[s] = x[:, i]
    sb.relplot(data, kind = "line", legend = columns)

def plot_graph(states: np.array, graph : np.array, classes=None, layout=None):

    graph = [(graph[0, i], graph[1, i]) for i in range(len(graph[0]))]

    g = nx.Graph()
    g.add_nodes_from(range(len(states)))
    g.add_edges_from(graph)
    color_map = {
        0:'red',
        1:'blue',
        2:'green',
        3:'yellow',
        4:'purple'
    }
    colors = []
    labels = {}
    for node in states:
        colors.append(color_map[node])
        labels[node] = classes[node]

    plt.figure(figsize=(10,10))

    if layout is not None:
        nx.draw(g, pos=layout, node_color=colors, font_size=10, node_size=15)
    else:
        pos = nx.spring_layout(g, seed=42)
        nx.draw(g, pos = pos, node_color=colors, font_size=10, node_size=25)
    # 创建图例的图标
    patches = []
    for i, item in enumerate(color_map.items()):
        if i>=len(classes):
            break
        patches.append(mpatches.Patch(color=item[1], label=classes[item[0]]))

    # 添加图例
    plt.legend(handles=patches)
    plt.show()

    return pos