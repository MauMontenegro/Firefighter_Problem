import networkx as nx
import random
import scipy.sparse as ss
import numpy as np
from ete3 import Tree
import matplotlib.pyplot as plt


def TreeConstruct(F, all_nodes, Tree):
    levels = nx.single_source_shortest_path_length(Tree, 0)
    max_level = max(levels.values())
    edges = list(nx.bfs_edges(Tree, source=0))

    # Add root
    F.add_child(name=str(0))
    all_nodes[0] = {}
    all_nodes[0]['level'] = levels[0]
    for edge in edges:
        all_nodes[str(edge[1])] = {}
        all_nodes[str(edge[1])]['level'] = levels[edge[1]]
        father = F.search_nodes(name=str(edge[0]))[0]
        child = str(edge[1])
        father.add_child(name=child)

    return F, all_nodes, max_level


def rndtree(config, path, file):
    # Create a Random Tree with N nodes
    N = config['experiment']['Size']  # Number of Nodes
    seed = config['experiment']['Seed']  # Experiment Seed
    Tree_netx = nx.random_tree(n=N, seed=seed)  # Create Random Tree
    all_nodes = {}  # List of All Nodes

    # Create Random Integer Edge Weights between intervals for Tree
    for (u, v, w) in Tree_netx.edges(data=True):
        w['weight'] = random.randint(config['experiment']['wr_l'], config['experiment']['wr_u'])

    # Obtain Adjacency Matrix in form of numpy array
    A = nx.to_numpy_array(Tree_netx)
    A = A.astype(int)

    # Original Adjacency Matrix of Tree
    O = nx.to_numpy_array(Tree_netx)
    O = O.astype(int)

    # Fill Adjacency Matrix to obtain all distances between Nodes (Non-metric distances)
    for row in range(0, N):
        for column in range(0, N):
            if row != column and A[row][column] == 0:
                A[row][column] = random.randint(config['experiment']['wr_l'], config['experiment']['wr_u'])

    # Create a Symmetric Matrix with upper part of A (For symmetric distances)
    A = np.triu(A) + np.tril(A.T)

    # Seed for reproducible layout
    pos = nx.spring_layout(Tree_netx, seed=seed)
    labels = nx.get_edge_attributes(Tree_netx, 'weight')
    nx.draw(Tree_netx, pos, with_labels=True)
    nx.draw_networkx_edge_labels(Tree_netx, pos, edge_labels=labels)

    # Saving Graph Image
    file_path = path + file + '.png'
    plt.savefig(file_path, format="PNG")

    # Initialize Tree Structure using ete3
    F = Tree()  # Initialize a Forest. Tree() is a func in ete3 with Newick format.

    # Construct Tree structure
    F, all_nodes, max_level = TreeConstruct(F, all_nodes, Tree_netx)

    # Variables
    env_update = config['experiment']['Env_Update']  # Update ratio of environment respect to agent
    time = max_level * env_update  # Budget Time before Tree burns entirely
    max_budget = max_level * env_update

    # Choose Initial Position of Agent
    initial_pos = config['experiment']['Initial_pos']

    # Create Additional config array due to environment differences
    Config = [config['experiment']['env_type'], config['experiment']['env_metric'], A]

    return [initial_pos, initial_pos], all_nodes, F, time, max_budget, Config
