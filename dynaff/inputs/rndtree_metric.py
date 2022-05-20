import random as rnd
import numpy as np
from ete3 import Tree
import matplotlib.pyplot as plt
import json
import os
import networkx as nx


class ExperimentLog:
    """
    A class used to save experiment information

    Attributes
    ----
    path: str
        formatted string that contains the path of saved instance
    file_name: str
        formatted string that contains the name of saved instance file

    Methods
    ---
    log_save(stats)
        Save .json file containing instance parameters
    """
    def __init__(self, path, file_name):
        """
        :param path: str
             formatted string that contains t
        :param file_name: str
            formatted string that contains the name of saved instance file
        """
        self.path = path
        self.file = file_name + '.json'
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.full_path = os.path.join(self.path, self.file)

    def log_save(self, stats):
        with open(self.full_path, 'w') as fp:
            json.dump(stats, fp)


def TreeConstruct(F, all_nodes, Tree, root):
    """Use networkx tree structure to create an ete3 Tree structure

    This function use node structure in a networkx Tree to progressively add nodes to a ete3 Tree using
    the fire node as root for the new tree. Also creates a list with nodes along his level in the new tree.

    :param F: Initial Tree structures in ete3 format
    :param all_nodes: List of all nodes in original Tree
    :param Tree: networkx Tree
    :param root: Initial fire node
    :return:
        F: Complete Tree in ete3 format with fire node as root
        all_nodes: List of all nodes in original Tree
        max_level: Level of rooted Tree
        levels: list of nodes with his level in rooted Tree
    """
    levels = nx.single_source_shortest_path_length(Tree, root)  # Level of nodes in rooted Tree
    degrees = list(Tree.degree)                                 # Degree of each node in rooted Tree
    max_level = max(levels.values())                            # Level of Tree
    edges = list(nx.bfs_edges(Tree, source=root))               # List of edges in rooted Tree with BFS order

    # Add root to ete3 rooted Tree structure
    F.add_child(name=str(root))
    all_nodes[str(root)] = {}
    all_nodes[str(root)]['level'] = levels[root]

    # Append nodes wit his level and add node to ete3 Tree
    # As 'edges' return edges in bfs order we can always search for 'father' in 'T'
    for edge in edges:
        all_nodes[str(edge[1])] = {}
        all_nodes[str(edge[1])]['level'] = levels[edge[1]]
        father = F.search_nodes(name=str(edge[0]))[0]
        child = str(edge[1])
        father.add_child(name=child)

    degrees.pop(-1)                                             # Pop agent position, as this is not on original graph.
    # Add degree for each node in his dictionary
    # (This will help us in some heuristics)
    for degree in degrees:
        all_nodes[str(degree[0])]['degree'] = degree[1]

    return F, all_nodes, max_level, levels


def DrawingInstance(layout, T, fire, N, path, file):
    """ Draw networkx Tree with current layout and node types

    :param layout: Tree nodes layout positions
    :param T: networkx Tree with agent position included
    :param fire: initial fire node
    :param N: Number of nodes
    :param path: path to save images
    :param file: name of file
    :return:
        List of actual burned nodes at t=0
        List of unlabeled nodes at t=0
    """
    # This is for get max and min labels in plot
    max_x_value = max(d[0] for d in layout.values())
    min_x_value = min(d[0] for d in layout.values())
    max_y_value = max(d[1] for d in layout.values())
    min_y_value = min(d[1] for d in layout.values())

    # Leaving only unlabeled nodes
    remaining_nodes = list(T.nodes)
    remaining_nodes.pop(fire)
    burnt_nodes = [fire]
    remaining_nodes.pop(-1)

    # Drawing Nodes
    options = {"edgecolors": "tab:gray", "node_size": 500, "alpha": 1}
    nx.draw_networkx_nodes(T, layout, nodelist=burnt_nodes, node_color="tab:red", **options)
    nx.draw_networkx_nodes(T, layout, nodelist=[N], node_color="tab:blue", **options)
    nx.draw_networkx_nodes(T, layout, nodelist=remaining_nodes, node_color="tab:green", **options)

    # Drawing edges
    nx.draw_networkx_edges(T, layout, width=2.0, alpha=0.9)

    # Labels and scale plotting
    plt.xticks(np.arange(min_x_value, max_x_value, 1))
    plt.yticks(np.arange(min_y_value, max_y_value, 1))
    plt.xlabel('X-Position', fontdict=None, labelpad=20)
    plt.ylabel('Y-Position', fontdict=None, labelpad=20)

    # Plotting and saving Image
    file_path = path + file + '.png'
    graph = plt.gcf()
    graph.savefig(file_path, format="PNG")
    plt.close()

    return burnt_nodes, remaining_nodes


def rndtree_metric(config, path, file, n_nodes):
    """ Creates an instance of the problem with metric distances.

    Create a random 'networkx' tree object, adds an external 'agent' node, and fill his original adjacency matrix
    with escalated metric distances between all nodes. Also, saves instance parameters and draws initial graph
    configuration.

    Parameters
    ---------------
    :param config: dic
        A dictionary containing experiment configuration variables
    :param path: str
        Path to save instance
    :param file: str
        Contains the file name
    :param n_nodes: int
        Total number of nodes in graph
    :return:
        Initial_agent_position: int array
        Node_List: dic
        Forest: ete3
        time: int
        max_budget: int
        Config: Array containing environment or metric config
        Plotting: Array containing drawing config
    """

    # Internal Variables
    N = n_nodes  # Number of Nodes
    seed = config['experiment']['Seed']  # Experiment Seed
    scale = 1  # Scale of distances
    starting_fire = rnd.randint(0, N - 1)  # Starting Fire Node
    a_x_pos = rnd.uniform(-1, 1) * scale  # X-axis Position of Agent
    a_y_pos = rnd.uniform(-1, 1) * scale  # Y-axis Position of Agent
    T = nx.random_tree(n=N, seed=seed)  # Create a Random Tree (nx use a Prufer Sequence by default)
    pos = nx.spring_layout(T, seed=seed)  # Use a spring layout to draw nodes
    nx.write_adjlist(T, "AdjList.adjlist")  # Save Adjacency list
    T_Ad = np.zeros((N + 1, N + 1))  # Adjacency Matrix that contains distances for all nodes including agent
    all_nodes = {}  # Dictionary to store all nodes
    saved_nodes = []  # Array of saved nodes for Drawing
    env_update = config['experiment']['Env_Update']  # Update ratio of environment respect to agent
    initial_pos = N  # Put agent as last node in Graph
    instance = {}  # dictionary to save instance parameters

    # Fill Adjacency Matrix with escalated distances in layout (without agent)
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
                T_Ad[row][column] = dist * scale

    # Scale Position in Layout for coherence between plotting and matrix distances
    for element in pos:
        pos[element][0] = pos[element][0] * scale
        pos[element][1] = pos[element][1] * scale

    # Adding Agent Node to Full Adjacency Matrix (Agent is added to last row and column 'N')
    for node in range(0, N):
        x_1 = pos[node][0]
        x_2 = a_x_pos
        y_1 = pos[node][1]
        y_2 = a_y_pos
        dist = np.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)
        T_Ad[node][N] = dist

    # Create a Symmetric Matrix with upper part of T_Ad (For symmetric distances)
    T_Ad_Sym = np.triu(T_Ad) + np.tril(T_Ad.T)

    # Add Agent Node to Tree and add his escalated position
    T.add_node(N)
    pos[N] = [a_x_pos, a_y_pos]

    # Draw Instance
    # 'remaining_nodes': Unlabeled Nodes for Drawing
    # 'burnt_nodes': Burnt labeled nodes for drawing
    remaining_nodes, burnt_nodes = DrawingInstance(pos, T, starting_fire, N, path, file)

    # Initialize a Forest.
    # Tree() is a function in 'ete3' library with Newick format.
    F = Tree()

    # Construct Tree structures
    F, all_nodes, max_level, levels = TreeConstruct(F, all_nodes, T, starting_fire)
    time = max_level * env_update  # Budget Time before Tree burns entirely
    max_budget = max_level * env_update  # Max budget of time

    # Save Instance
    logger = ExperimentLog('instance', 'instance_info')  # Create Logger Class to store instance parameters
    nx.write_adjlist(T, "instance/MFF_Tree.adjlist")  # Saving Full Distance Matrix
    np.save("instance/FDM_MFFP.npy", T_Ad_Sym)  # Saving Numpy array full distance matrix
    instance['N'] = N
    instance['seed'] = seed
    instance['scale'] = scale
    instance['start_fire'] = starting_fire
    instance['a_pos_x'] = a_x_pos
    instance['a_pos_y'] = a_y_pos
    logger.log_save(instance)

    # Save position layout as .json file
    # .json file need a list format ot save
    for element in pos:
        pos[element] = list(pos[element])
    with open('instance/layout_MFF.json', 'w') as layout_file:
        layout_file.write(json.dumps(pos))
    layout_file.close()

    # Create Additional config array due to environments differences
    Config = [config['experiment']['env_type'], config['experiment']['env_metric'], T_Ad_Sym]

    # Create Plotting array containing drawing variables
    Plotting = [T, pos, burnt_nodes, remaining_nodes, N, path + file + '/', levels, saved_nodes]

    return [initial_pos, initial_pos], all_nodes, F, time, max_budget, Config, Plotting
