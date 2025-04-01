import torch
import numpy as np
import seaborn as sb
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_series(x : np.array, columns : list, fig_size=None):
    """
    Plots time series data from an array using Seaborn's line plot functionality. Each series
    is represented as a separate line on the plot.

    Parameters
    ----------
    x : np.array
        A NumPy array containing the data points of the time series.
    columns : list
        List of strings representing the column names or labels for each time series in the array.

    Returns
    -------
    None
        Displays the plot directly and returns None.
    """
    data = dict.fromkeys(columns)
    for i ,s in enumerate(columns):
        data[s] = x[:, i]
    if fig_size is not None:
        plt.figure(figsize=fig_size)
    sb.relplot(data, kind = "line", legend = columns)

def plot_graph(states: np.array, graph : np.array, classes=None, layout=None):
    """
    Plots a graph using NetworkX where nodes are colored based on their state and annotated with class labels.
    The graph's layout can be specified; otherwise, a spring layout is used.

    Parameters
    ----------
    states : np.array
        Array containing the state of each node used to determine the color of nodes.
    graph : np.array
        A 2D array where each column represents an edge between two nodes (start and end).
    classes : list, optional
        List of class labels corresponding to each node. Default is None.
    layout : dict, optional
        A dictionary defining the position of each node for custom layouts. If None, a spring layout is used. Default is None.

    Returns
    -------
    dict
        The positions of the nodes in the plot, useful for customizing the layout or for further graphical analysis.
    """
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

    patches = []
    for i, item in enumerate(color_map.items()):
        if i>=len(classes):
            break
        patches.append(mpatches.Patch(color=item[1], label=classes[item[0]]))


    plt.legend(handles=patches)
    plt.show()

    return pos
