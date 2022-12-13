import numpy as np
from ete3 import Tree
import matplotlib.pyplot as plt
import json
import os
import networkx as nx

# Visualization
from pyvis.network import Network

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
    root_degree=Tree.degree[root]
    degrees = list(Tree.degree)                                 # Degree of each node in rooted Tree
    max_degree = max(degrees,key=lambda item:item[1])[1]
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

    return F, all_nodes, max_level, levels, root_degree, max_degree

def write_FFP_file(dimension, edges, coords, fire, output_path):
    with open(output_path, "w") as writer:
        writer.write("DIMENSION: {}\n".format(dimension))
        writer.write("FIRE_START: {}\n".format(fire))
        writer.write("FIREFIGHTER: {}\n".format(dimension))

        # coord display
        writer.write("DISPLAY_DATA_SECTION\n")
        for i in range(dimension + 1):
            coord = coords[i]
            writer.write("{} {} {}\n".format(i, coord[0], coord[1]))

        # edge section
        writer.write("EDGE_SECTION\n")
        for e in edges:
            writer.write("{} {}\n".format(e[0], e[1]))

def write_FFP_summary(instance, output_file):
    with open(output_file, "w") as writer:
        writer.write("TREE SEED: {}\n".format(instance["seed"]))
        writer.write("DIMENSION: {}\n".format(instance["N"]))
        writer.write("FIRE_START: {}\n".format(instance["start_fire"]))
        writer.write("FIREFIGHTER: {}\n".format(instance["N"]))
        writer.write("DELTA: {}\n".format(instance["delta"]))
        writer.write("ROOT DEGREE: {}\n".format(instance["root_degree"]))
        writer.write("MAX DEGREE: {}\n".format(instance["max_degree"]))
        writer.write("TREE HEIGHT: {}\n".format(instance["tree_height"]))
        writer.write("SCALE_DISTANCE: {}\n".format(instance["scale"]))


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
    fig,ax=plt.subplots()
    options = {"edgecolors": "tab:gray", "node_size": 300, "alpha": 1}
    nx.draw(T,layout, with_labels=True,font_size=10,ax=ax)
    nx.draw_networkx_nodes(T, layout, nodelist=burnt_nodes, node_color='#e33434',label="Ignition", **options)
    nx.draw_networkx_nodes(T, layout, nodelist=[N], node_color='#34e3e0',label="Firefighter", **options)
    nx.draw_networkx_nodes(T, layout, nodelist=remaining_nodes, node_color='#62fa69',label="Remaining", **options)

    # Drawing edges
    nx.draw_networkx_edges(T, layout, width=1.0, alpha=0.8,edge_color='#959895')

    # Labels and scale plotting
    plt.axis('on')
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    #plt.xticks(np.arange(min_x_value, max_x_value, 1))
    #plt.yticks(np.arange(min_y_value, max_y_value, 1))
    plt.xlabel('X-Position', fontdict=None, labelpad=5)
    plt.ylabel('Y-Position', fontdict=None, labelpad=5)
    title_label= "MFP Tree Instance with {N} vertices".format(N=N)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.legend(loc="lower left", ncol=1, bbox_to_anchor=(1, 0.5), labelspacing=1)
    plt.title(title_label)
    #nt=Network('500px', '500px')
    #nt.from_nx(T)
    #nt.show('nx.html')

    # Plotting and saving Image
    file_path = path + file + '.png'
    graph = plt.gcf()
    graph.savefig(file_path, format="PNG")
    plt.close()
    return remaining_nodes, burnt_nodes


def rndtree_metric(config, path, file, n_nodes, rnd_generators):
    """ Creates N instances for a determined Size Tree with metric distances.

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
    :param rnd_generators: default_generator
        batch of N random number generators for each instance
    :return:
        Initial_agent_position: int array
        Node_List: dic
        Forest: ete3
        time: int
        max_budget: int
        Config: Array containing environment or metric config
        Plotting: Array containing drawing config
    """
    for instance in range(config['experiment']['instances']):
        instance_path = path + "Instance_" + str(instance) + "/" # Create Instance Specific Folder
        N = n_nodes
        n_instances = config['experiment']['instances'] # Number of instances per Node Size
        scale = config['experiment']['scale']  # Edge distance scale
        r_degree = config['experiment']['root_degree']  # Force Tree to have a root degree of this size
        env_update = config['experiment']['Env_Update']  # Update ratio of environment respect to agent
        delta = config['experiment']['delta']

        # Random Variables. Use rnd_generator specific to each instance.
        # Generate Only Trees with fire root node with desired degree
        starting_fire = rnd_generators[instance].integers(0, N - 1)
        rootd_check = True
        while rootd_check:
            tree_seed = rnd_generators[instance].integers(2 ** 32 - 1)
            T = nx.random_tree(n=N, seed=int(tree_seed))
            pos = nx.spring_layout(T, seed=int(tree_seed),scale=scale)  # Use a spring layout to draw nodes
            if T.degree[starting_fire] == r_degree:
                rootd_check = False
        # Limit agent distance from root
        limit_agent_radius_inf = delta[0] * scale
        limit_agent_radius_sup = delta[1] * scale

        # Needed Structures
        T_Ad = np.zeros((N + 1, N + 1))  # Adjacency Matrix that contains distances for all nodes including agent
        all_nodes = {}  # Dictionary to store all nodes
        saved_nodes = []  # Array of saved nodes for Drawing
        initial_pos = N  # Put agent as last node in Graph
        instance_ = {}  # dictionary to save instance parameters

        # Save Instance
        logger = ExperimentLog(instance_path, 'instance_info')  # Create Logger Class to store instance parameters

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
                    T_Ad[row][column] = np.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)

        # Force agent to be at limit distance from ignition vertex
        ref_x, ref_y = pos[starting_fire][0],pos[starting_fire][1]  # Get ignition vertex position reference
        x_offset = rnd_generators[instance].uniform(limit_agent_radius_inf, limit_agent_radius_sup)
        if rnd_generators[instance].random() < 0.5:
            x_offset = x_offset * -1
        y_offset = rnd_generators[instance].uniform(limit_agent_radius_inf, limit_agent_radius_sup)
        if rnd_generators[instance].random() < 0.5:
            y_offset = y_offset * -1
        a_x_pos = ref_x + x_offset
        a_y_pos = ref_y + y_offset

        # Adding Agent Node to Full Adjacency Matrix (Agent is added to last row and column 'N')
        for node in range(0, N):
            x_1 = pos[node][0]
            x_2 = a_x_pos
            y_1 = pos[node][1]
            y_2 = a_y_pos
            T_Ad[node][N] = np.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)

        # Create a Symmetric Matrix with upper part of T_Ad (For symmetric distances)
        T_Ad_Sym = np.triu(T_Ad) + np.tril(T_Ad.T)

        # Add Agent Node to Tree and add his escalated position
        T.add_node(N)
        pos[N] = [a_x_pos, a_y_pos]

        # Draw Instance
        # 'remaining_nodes': Unlabeled Nodes for Drawing
        # 'burnt_nodes': Burnt labeled nodes for drawing
        remaining_nodes, burnt_nodes = DrawingInstance(pos, T, starting_fire, N, instance_path, file)

        # Initialize a Forest.
        # Tree() is a function in 'ete3' library with Newick format.
        F = Tree()

        #TODO: Instance Generator does not need to generate Tree

        # Construct Tree structures
        F, all_nodes, max_level, levels, root_degree, max_degree = TreeConstruct(F, all_nodes, T, starting_fire)
        # time = max_level * env_update  # Budget Time before Tree burns entirely
        # max_budget = max_level * env_update  # Max budget of time


        nx.write_adjlist(T, instance_path +'MFF_Tree.adjlist')  # Saving Full Distance Matrix
        np.save(instance_path + "FDM_MFFP.npy", T_Ad_Sym)  # Saving Numpy array full distance matrix

        # Instance Variables
        instance_['N'] = N
        instance_['seed'] = int(tree_seed)
        instance_['scale'] = scale
        instance_['start_fire'] = int(starting_fire)
        instance_['a_pos_x'] = int(a_x_pos)
        instance_['a_pos_y'] = int(a_y_pos)
        instance_['tree_height'] = max_level
        instance_['root_degree'] = root_degree
        instance_['max_degree'] = max_degree
        # Initial distance between agent position and fire ignition
        instance_['delta'] = T_Ad[starting_fire][N]
        logger.log_save(instance_)

        # Save position layout as .json file
        # .json file need a list format ot save
        for element in pos:
            pos[element] = list(pos[element])
        with open(instance_path + 'layout_MFF.json', 'w') as layout_file:
            layout_file.write(json.dumps(pos))
        layout_file.close()

        # File created for Backtracking Algorithm
        output_file = instance_path + "BCKTRCK.mfp"
        write_FFP_file(n_nodes, T.edges(), pos, starting_fire, output_file)

        # CREATE A SUMMARY FILE FOR EACH INSTANCE
        output_file = instance_path + "SUMMARY.mfp"
        write_FFP_summary(instance_, output_file)

        # Create Additional config array due to environments differences
        #Config = [config['experiment']['env_type'], config['experiment']['env_metric'], T_Ad_Sym]

        # Create Plotting array containing drawing variables
        #Plotting = [T, pos, burnt_nodes, remaining_nodes, N, instance_path + file + '/', levels, saved_nodes]

        #return [initial_pos, initial_pos], all_nodes, F, time, max_budget, Config, Plotting, pos
