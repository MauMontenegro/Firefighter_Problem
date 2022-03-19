import networkx as nx
import random as rnd
import numpy as np
from ete3 import Tree
import matplotlib.pyplot as plt
from sklearn.manifold import SpectralEmbedding

def TreeConstruct(F, all_nodes, Tree):
    levels = nx.single_source_shortest_path_length(Tree, 0)
    degrees = list(Tree.degree)
    max_level = max(levels.values())
    edges = list(nx.bfs_edges(Tree, source=0))

    # Add root
    F.add_child(name=str(0))
    all_nodes[str(0)] = {}
    all_nodes[str(0)]['level'] = levels[0]

    for edge in edges:
        all_nodes[str(edge[1])] = {}
        all_nodes[str(edge[1])]['level'] = levels[edge[1]]
        father = F.search_nodes(name=str(edge[0]))[0]
        child = str(edge[1])
        father.add_child(name=child)
    print(all_nodes)
    for degree in degrees:
        all_nodes[str(degree[0])]['degree'] = degree[1]

    return F, all_nodes, max_level


def rndtree_metric(config, path, file, n_nodes):
    N = n_nodes  # Number of Nodes
    seed = config['experiment']['Seed']  # Experiment Seed

    # Create a Random Tree (nx use a Prufer Sequence) and get pos layout of nodes
    T = nx.random_tree(n=N, seed=seed)
    pos = nx.spring_layout(T, seed=seed)

    T_Ad = np.zeros((N, N))
    for row in range(0, N):
        for column in range(row, N):
            if row == column:
                T_Ad[row][column] = 0
            else:
                x_1 = pos[row][0]
                x_2 = pos[column][0]
                y_1 = pos[row][1]
                y_2 = pos[column][1]
                dist = np.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)
                T_Ad[row][column] = dist * 10

    # Create a Symmetric Matrix with upper part of A_Emb (For symmetric distances)
    T_Ad_Sym = np.triu(T_Ad) + np.tril(T_Ad.T)

    all_nodes = {}  # List of All Nodes

    # Seed for reproducible layout
    nx.draw(T, pos, with_labels=True)

    # Saving Graph Image
    file_path = path + file + '.png'
    plt.savefig(file_path, format="PNG")
    plt.close()

    # Initialize Tree Structure using ete3
    F = Tree()  # Initialize a Forest. Tree() is a func in ete3 with Newick format.

    # Construct Tree structure
    F, all_nodes, max_level = TreeConstruct(F, all_nodes, T)

    # Variables
    env_update = config['experiment']['Env_Update']  # Update ratio of environment respect to agent
    time = max_level * env_update  # Budget Time before Tree burns entirely
    max_budget = max_level * env_update

    # Choose Initial Position of Agent
    initial_pos = config['experiment']['Initial_pos']

    # Create Additional config array due to environment differences
    Config = [config['experiment']['env_type'], config['experiment']['env_metric'], T_Ad_Sym]

    return [initial_pos, initial_pos], all_nodes, F, time, max_budget, Config
