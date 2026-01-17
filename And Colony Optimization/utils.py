import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def generate_graph(n):
    graph = np.zeros(shape=(n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            cost = random.randint(1, 50)
            graph[i, j] = cost
            graph[j, i] = cost

    return graph

def show_graph_with_labels(adjacency_matrix, n):
    labels = {}
    for i in range(n):
        labels[i] = str(i)
    rows, cols = np.where(adjacency_matrix != 0)
    gr = nx.Graph()
    for r in rows:
        for c in cols:
            if r != c:
                gr.add_edge(r, c, w=adjacency_matrix[c][r])
                gr.add_edge(c, r, w=adjacency_matrix[c][r])

    pos = nx.spring_layout(gr)
    nx.draw(gr, node_size=500, labels=labels, with_labels=True)
    edge_labels=nx.get_edge_attributes(gr,'w')
    nx.draw_networkx_edge_labels(gr, pos, edge_labels = edge_labels)
    plt.show()

def show_graph_with_labels_and_path(adjacency_matrix, n, shortest_path):
    labels = {}
    for i in range(n):
        labels[i] = str(i)
    rows, cols = np.where(adjacency_matrix != 0)
    gr = nx.Graph()
    for r in rows:
        for c in cols:
            if r != c:
                gr.add_edge(r, c, w=adjacency_matrix[c][r])
                gr.add_edge(c, r, w=adjacency_matrix[c][r])

    red_edges = [(shortest_path[i], shortest_path[i + 1]) if shortest_path[i] < shortest_path[i+1] else (shortest_path[i + 1], shortest_path[i])  
                    for i in range(len(shortest_path) - 1)]

    edge_col = ['black' if not edge in red_edges else 'red' for edge in gr.edges()]
    print(edge_col)
    pos = nx.spring_layout(gr)
    nx.draw(gr, node_size=500, edge_color= edge_col, labels=labels, with_labels=True)
    edge_labels=nx.get_edge_attributes(gr,'w')
    nx.draw_networkx_edge_labels(gr, pos, font_color= edge_col, edge_labels = edge_labels)
    plt.show()
