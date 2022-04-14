import networkx as nx
import random as rnd
import numpy as np
from ete3 import Tree
import matplotlib.pyplot as plt


def TreeConstruct(F, all_nodes, Tree, root):
    levels = nx.single_source_shortest_path_length(Tree, root)
    degrees = list(Tree.degree)
    max_level = max(levels.values())
    edges = list(nx.bfs_edges(Tree, source=root))

    # Add root
    F.add_child(name=str(root))
    all_nodes[str(root)] = {}
    all_nodes[str(root)]['level'] = levels[root]

    for edge in edges:
        all_nodes[str(edge[1])] = {}
        all_nodes[str(edge[1])]['level'] = levels[edge[1]]
        father = F.search_nodes(name=str(edge[0]))[0]
        child = str(edge[1])
        father.add_child(name=child)

    degrees.pop(-1)  # Pop agent position, as this is not rom original graph.
    for degree in degrees:
        all_nodes[str(degree[0])]['degree'] = degree[1]

    return F, all_nodes, max_level, levels


def rndtree_metric(config, path, file, n_nodes):
    N = n_nodes  # Number of Nodes
    seed = config['experiment']['Seed']  # Experiment Seed
    scale = 10
    starting_fire = rnd.randint(0, N - 1)

    print('Starting fire in Node:')
    print(starting_fire)

    # Adding Agent Node
    a_x_pos = rnd.uniform(-1, 1) * scale
    a_y_pos = rnd.uniform(-1, 1) * scale

    # Create a Random Tree (nx use a Prufer Sequence) and get 'pos' layout for nodes
    T = nx.random_tree(n=N, seed=seed)
    # Could use spring or spectral Layout
    pos = nx.spring_layout(T, seed=seed)
    nx.write_adjlist(T, "AdjList.adjlist")

    T_Ad = np.zeros((N + 1, N + 1))

    # Create Adjacency Matrix with escalated distances in layout
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
                T_Ad[row][column] = dist * scale  # Scale factor of 10

    # Scale Position for plotting
    for element in pos:
        pos[element][0] = pos[element][0] * scale
        pos[element][1] = pos[element][1] * scale

    # Adding Agent Node to Full Adjacency Matrix
    for node in range(0, N):
        x_1 = pos[node][0]
        x_2 = a_x_pos
        y_1 = pos[node][1]
        y_2 = a_y_pos
        dist = np.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)
        T_Ad[node][N] = dist

    # Create a Symmetric Matrix with upper part of T_Ad (For symmetric distances)
    T_Ad_Sym = np.triu(T_Ad) + np.tril(T_Ad.T)

    # Add Agent Node to Tree and add his scalated position
    T.add_node(N)
    pos[N] = [a_x_pos, a_y_pos]

    all_nodes = {}  # List of All Nodes

    # Draw and saving Graph

    # This is for get max and min labels in plotting
    max_x_value = max(d[0] for d in pos.values())
    min_x_value = min(d[0] for d in pos.values())
    max_y_value = max(d[1] for d in pos.values())
    min_y_value = min(d[1] for d in pos.values())

    # Leaving only unlabeled nodes
    remaining_nodes = list(T.nodes)
    remaining_nodes.pop(starting_fire)
    burnt_nodes = [starting_fire]
    remaining_nodes.pop(-1)
    saved_nodes=[]

    options = {"edgecolors": "tab:gray", "node_size": 500, "alpha": 1}
    nx.draw_networkx_nodes(T, pos, nodelist=burnt_nodes, node_color="tab:red", **options)
    nx.draw_networkx_nodes(T, pos, nodelist=[N], node_color="tab:blue", **options)
    nx.draw_networkx_nodes(T, pos, nodelist=remaining_nodes, node_color="tab:green", **options)

    nx.draw_networkx_edges(T, pos, width=2.0, alpha=0.9)

    plt.xticks(np.arange(min_x_value, max_x_value, 1))
    plt.yticks(np.arange(min_y_value, max_y_value, 1))
    plt.xlabel('X-Position', fontdict=None, labelpad=20)
    plt.ylabel('Y-Position', fontdict=None, labelpad=20)

    file_path = path + file + '.png'
    graph = plt.gcf()
    graph.savefig(file_path, format="PNG")
    plt.close()

    # Initialize Tree Structure using ete3
    F = Tree()  # Initialize a Forest. Tree() is a func in ete3 with Newick format.

    # Construct Tree structure
    F, all_nodes, max_level, levels = TreeConstruct(F, all_nodes, T, starting_fire)

    # Variables
    env_update = config['experiment']['Env_Update']  # Update ratio of environment respect to agent
    time = max_level * env_update  # Budget Time before Tree burns entirely
    max_budget = max_level * env_update
    print('Timee')
    print(time)
    # Choose Initial Position of Agent
    initial_pos = N

    #Saving Full Distance Matrix
    f = open("Full_Matrix.txt", "w")
    f.write(str(T_Ad_Sym))

    # Create Additional config array due to environment differences
    Config = [config['experiment']['env_type'], config['experiment']['env_metric'], T_Ad_Sym]

    Plotting = [T, pos, burnt_nodes, remaining_nodes, N, path + file + '/', levels, saved_nodes]

    return [initial_pos, initial_pos], all_nodes, F, time, max_budget, Config, Plotting
