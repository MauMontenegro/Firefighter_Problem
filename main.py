import gym
import gym_cellular_automata as gymca
import matplotlib.pyplot as plt
import csv
from ete3 import Tree

ProtoEnv = gymca.prototypes[1]
env = ProtoEnv(nrows=20, ncols=20)

obs = env.reset()

total_reward = 0.0
done = False
step = 0
threshold = 60


def labeling():
    all_nodes = {}
    with open('arbirary_grid') as file:
        reader = csv.reader(file)
        for row in reader:
            label = '{x}-{y} '.format(x=row[0], y=row[1])
            all_nodes[label] = {}
            all_nodes[label]['visited'] = False
    return all_nodes


def TreeConstruct(all_nodes, t):
    for tree_node in all_nodes:
        # print('Analyzing node:{}'.format(tree_node))
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
    print(all_nodes)


def SavedNodes(t, cutting_node):
    # First we get the corresponding branch of T
    saved = 0
    father = t.search_nodes(name=cutting_node)[0]

    for node in father.iter_descendants("postorder"):
        saved += 1
    #print("Saved:{}".format(saved))

    # Now, we detach this sub_tree from original
    #father.detach()
    # Observe remaining Tree and return saved nodes of this cut node
    #print(t.get_ascii(show_internal=True))
    return saved

def DPSolution(a_pos,time_budget,update,all_nodes,burn_tree):
    # time_budget are the max level =8
    # update are the steps that agent can act before env changes

    # Adding Base Cases for our Matrix
    for node in all_nodes:
        all_nodes[node]['time']=[0,0]

    for time in range(2, time_budget):
        actual_level = time_budget-time
        for node in all_nodes:
            if all_nodes[node]['level'] == actual_level:
                saved = SavedNodes(burn_tree, node)
                all_nodes[node]['time'].append(saved)
            else:
                all_nodes[node]['time'].append(0)

    # Construct Solution
    memoization=[]
    for node in all_nodes:
        memoization.append(all_nodes[node]['time'])

    for i in range(0,7):

    print(all_nodes)
    print(memoization)



if __name__ == '__main__':
    # 1. Labeling nodes in active burning graph
    all_nodes = labeling()

    # 2. Create Tree Structure using Newick format
    grid = obs[0]
    print(obs)
    agent_pos_x, agent_pos_y = obs[1][1][0], obs[1][1][1]

    t = Tree()  # Initialize a tree. Tree() is a func in ete3 for Newick format.

    # 3. Create a Rooted Tree
    node_name = "0-0 "  # Add root as the first node in Tree
    t.add_child(name=node_name)
    all_nodes[node_name] = {}
    all_nodes[node_name]['level'] = 0
    all_nodes[node_name]['visited'] = True  # Change status of root as visited

    # 4. Looking for Tree cells in Neighborhood for all nodes in Burning Graph
    TreeConstruct(all_nodes, t)

    # 5. Checking and Saving
    t.write(format=1, outfile="burning_tree.nw")  # Saving Original Tree
    print(t.get_ascii(show_internal=True))  # Printing Tree Structure


    # Dynamic Programming Algorithm
    DPSolution([agent_pos_x,agent_pos_y],8,2,all_nodes,t)

# TO DO
    # Greedy Strategies: Mas cercano que salve algo
    # Nodo valido que salve más arboles
    # Proponer mas estrategias greeedy
    # Programar DP con Top-Down y Hash tables con arboles como keys

