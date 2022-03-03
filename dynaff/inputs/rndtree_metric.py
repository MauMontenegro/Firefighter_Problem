import networkx as nx
import random
import numpy as np
from ete3 import Tree
import matplotlib.pyplot as plt
from sklearn.manifold import SpectralEmbedding

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


def rndtree_metric(config, path, file):
    N = config['experiment']['Size']  # Number of Nodes
    seed = config['experiment']['Seed']  # Experiment Seed
    # Create a Random Graph with N nodes
    G = nx.gnp_random_graph(n=N, p=0.4, seed=seed)

    # Create Random Integer Edges Between nodes (No metric Distance)
    for (u, v, w) in G.edges(data=True):
        w['weight'] = random.randint(config['experiment']['wr_l'], config['experiment']['wr_u'])

    # Getting Adjacency Matrix from G
    A = nx.to_numpy_array(G)

    # Make an Embedding in Plane with metric distances between nodes.
    embedding = SpectralEmbedding(n_components=2)
    Emb = embedding.fit_transform(A)

    # Construct New Adjacency From Embedding (Metric Edge values)
    A_Emb = np.zeros((N, N))
    for row in range(0, N):
        for column in range(row, N):
            if row == column:
                A_Emb[row][column] = 0
            else:
                A_Emb[row][column] = np.linalg.norm(Emb[row] - Emb[column]) * 10

    # Create a Symmetric Matrix with upper part of A_Emb (For symmetric distances)
    A_Emb = np.triu(A_Emb) + np.tril(A_Emb.T)

    # Induce a Tree using BFS on G
    BFS_Tree = nx.minimum_spanning_tree(G)  # Create Random Tree
    all_nodes = {}  # List of All Nodes

    # Seed for reproducible layout
    pos = nx.spring_layout(BFS_Tree, seed=seed)
    labels = nx.get_edge_attributes(BFS_Tree, 'weight')
    nx.draw(BFS_Tree, pos, with_labels=True)

    # Saving Graph Image
    file_path = path + file + '.png'
    plt.savefig(file_path, format="PNG")

    # Initialize Tree Structure using ete3
    F = Tree()  # Initialize a Forest. Tree() is a func in ete3 with Newick format.

    # Construct Tree structure
    F, all_nodes, max_level = TreeConstruct(F, all_nodes, BFS_Tree)

    # Variables
    env_update = config['experiment']['Env_Update']  # Update ratio of environment respect to agent
    time = max_level * env_update  # Budget Time before Tree burns entirely
    max_budget = max_level * env_update

    # Choose Initial Position of Agent
    initial_pos = config['experiment']['Initial_pos']

    # Create Additional config array due to environment differences
    Config = [config['experiment']['env_type'], config['experiment']['env_metric'], A_Emb]

    return [initial_pos, initial_pos], all_nodes, F, time, max_budget, Config
