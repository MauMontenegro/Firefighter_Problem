import gym
import gym_cellular_automata as gymca
import csv
from ete3 import Tree


def labeling():
    # In arbitrary_grid first node must be the fire root
    all_nodes = {}
    root = []
    c = 0
    with open('arbitrary_grid2') as file:
        reader = csv.reader(file)
        for row in reader:
            label = '{x}-{y} '.format(x=row[0], y=row[1])
            if c == 0:
                root.append(label)
            all_nodes[label] = {}
            all_nodes[label]['visited'] = False
            c += 1
    return all_nodes, root[0], c


def TreeConstruct(all_nodes, t, grid):
    for tree_node in all_nodes:
        node_name = tree_node
        level = all_nodes[node_name]['level']
        n_x, n_y = tree_node.split("-")
        n_x = int(n_x)
        n_y = int(n_y)
        father = t.search_nodes(name=node_name)[0]
        for i in range(n_x - 1, n_x + 2):
            for j in range(n_y - 1, n_y + 2):
                if 0 <= i <= 20 and 0 <= j <= 20:
                    if grid[i][j] == 3:  # Looking for adjacent tree cells
                        node_name = '{x}-{y} '.format(x=str(i), y=str(j))
                        if all_nodes[node_name]['visited'] == False:  # This node is not already visited
                            all_nodes[node_name]['visited'] = True  # Mark as visited
                            all_nodes[node_name]['level'] = level + 1  # Mark as visited
                            father.add_child(name=node_name)  # Add node as a child
    #print(all_nodes)


def caenv(config):
    # Initial Env Config
    ProtoEnv = gymca.prototypes[1]
    N = config['experiment']['Size']
    env = ProtoEnv(nrows=N, ncols=N)
    obs = env.reset()
    grid = obs[0]
    agent_pos_x, agent_pos_y = obs[1][1][0], obs[1][1][1]

    # 1. Labeling nodes in active burning graph
    all_nodes, root_label, n_nodes = labeling()
    # all_nodes : Dictionary of all nodes in burning graph
    # root_label: label of burning root
    # n_nodes: Number of nodes in Graph

    # 2. Create Tree Structure using Newick format
    # http://etetoolkit.org/docs/latest/tutorial/tutorial_trees.html
    F = Tree()  # Initialize a Forest. Tree() is a func in ete3 for Newick format.

    # 3. Create a Rooted Tree
    F.add_child(name=root_label)
    all_nodes[root_label] = {}
    all_nodes[root_label]['level'] = 0  # As it is a root then level in Tree is 0
    all_nodes[root_label]['visited'] = True  # Change status of root as visited

    # 4. Build Forest Structure starting in root
    TreeConstruct(all_nodes, F, grid)

    # 5. Checking and Saving
    # t.write(format=1, outfile="burning_tree.nw")  # Saving Original Tree
    #print(F.get_ascii(show_internal=True))  # Printing Tree Structure

    # Variables
    tree_levels = max(int(d['level']) for d in all_nodes.values())
    env_update = 2  # Update ratio of environment respect to agent (2:1)
    time = tree_levels * env_update  # Budget Time before Tree burns entirely
    max_budget = tree_levels * env_update
    Hash = {}  # Here, we store a Forest with specific conditions and his max value of saved trees

    # Dynamic Programming Algorithm
    all_nodes.pop(root_label)  # We dont want to count our fire root in algorithm

    # Create Additional config list due to environment differences

    Config = [config['experiment']['env_type'], grid]

    return [agent_pos_x, agent_pos_y], all_nodes, F, time, Hash, max_budget, Config
