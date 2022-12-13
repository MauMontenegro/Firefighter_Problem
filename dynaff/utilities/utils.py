import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import json

def ComputeTime(a_pos, node_pos,dist_matrix):
    Adj = dist_matrix
    delta_time = Adj[int(node_pos)][int(a_pos)]
    return delta_time

def DetachNode(t, cutting_node, levels=0,saved_nodes=0):
    saved = 1
    father = t.search_nodes(name=cutting_node)[0]
    for node in father.iter_descendants("postorder"):
        saved += 1
        #levels.pop(int(node.name))
        #saved_nodes.append(int(node.name))
    father.detach()
    return saved

def Find_Solution(Forest, Time, a_pos, Hash, dist_matrix):
    # Construct the Key for Hash ( String: "Forest;time;pos_x,pos_y" )
    frame = 0
    total_budget = Time
    #spread_time = update
    #last_level_burnt = 0
    elapsed = 0

    key = Forest.write(format=8) + ';' + str(Time) + ';' + str(a_pos)
    #graphSolution(plotting,frame)
    Solution = []
    while Hash[key]['value'] != 0:
        frame += 1
        # Getting node of max value
        # print('Times to all nodes:')
        # print(config[2][plotting[4],:])
        node = Hash[key]['max_node']

        # Computes Remaining Time if agent travel to this node
        Time = Time - ComputeTime(a_pos, node, dist_matrix)
        elapsed += (total_budget-Time)

        #all_levels = plotting[6]
        #levels_to_burnt = int((total_budget - Time)/spread_time) + last_level_burnt
        #print('Levels to Burnt')
        #print(levels_to_burnt)
        # for level in range(last_level_burnt+1,(last_level_burnt + levels_to_burnt)+1):
        #     print('Level')
        #     print(level)
        #     keys = [k for k, v in all_levels.items() if v <= level]
        #     print('Keys')
        #     print(keys)
        #     # Assign to burning nodes and quit from remaining
        #     for element in keys:
        #         if element not in plotting[2]:
        #             plotting[2].append(element)
        #         if element in plotting[3]:
        #             plotting[3].remove(element)
        #     graphSolution(plotting, frame, 0, level, spread_time*level)
        #     frame += 1

        # Change pos of agent to next node in solution
        #pos = plotting[1]

        #pos[plotting[4]] = pos[int(node)]
        #plotting[1] = pos
        # Saved Trees by selecting this node
        saved = DetachNode(Forest, node)

        # last_level_burnt += levels_to_burnt
        # Add saved Node
        # plotting[7].append(int(node))

        # New agent position moves to node position
        a_pos = node
        Solution.append(node)
        # New Key
        #graphSolution(plotting, frame,total_budget-Time,levels_to_burnt,elapsed)
        total_budget = Time
        key = Forest.write(format=8) + ';' + str(Time) + ';' + str(a_pos)
    return Solution


def graphSolution(plot_config,frame_num,spend_time=0,burned_levels=0,total_elapsed=0):
    Tree = plot_config[0]
    pos = plot_config[1]
    burnt_nodes = plot_config[2]
    remaining_nodes = plot_config[3]
    agent_pos = plot_config[4]
    path = plot_config[5]
    saved_nodes= plot_config[7]

    # This is for get max and min labels in plotting
    max_x_value = max(d[0] for d in pos.values())
    min_x_value = min(d[0] for d in pos.values())
    max_y_value = max(d[1] for d in pos.values())
    min_y_value = min(d[1] for d in pos.values())

    # Nodes
    options = {"edgecolors": "tab:gray", "node_size": 500, "alpha": 1}
    nx.draw_networkx_nodes(Tree, pos, nodelist=burnt_nodes, node_color="tab:red", **options)
    nx.draw_networkx_nodes(Tree, pos, nodelist=remaining_nodes, node_color="tab:green", **options)
    nx.draw_networkx_nodes(Tree, pos, nodelist=saved_nodes, node_color="tab:blue", **options)
    nx.draw_networkx_nodes(Tree, pos, nodelist=[agent_pos], node_color="tab:gray", **options)

    # Edges
    nx.draw_networkx_edges(Tree, pos, width=2.0, alpha=0.9)
    plt.xticks(np.arange(min_x_value, max_x_value, 1))
    plt.yticks(np.arange(min_y_value, max_y_value, 1))
    plt.xlabel('X-Position', fontdict=None, labelpad=20)
    plt.ylabel('Y-Position', fontdict=None, labelpad=20)

    scal=10
    plt.text(scal*-.8, scal*-1.3,"Travel time: {0:.2f}".format(spend_time))
    plt.text(scal*-.8, scal*-1.4, "Burned Levels: {l}".format(l=burned_levels))
    plt.text(scal*-.8, scal*-1.5, "Total elapsed Time: {0:.2f}".format(total_elapsed))

    graph = plt.gcf()
    frame = str(frame_num)
    full_path = path + "/"
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    graph.savefig(full_path + 'frame_' + frame, format="PNG")
    plt.close()


def SavedNodes(t, cutting_node, Valid_):
    # First we get the corresponding branch of T
    saved = 1 # As we "defend nodes" detach node is also saved
    father = t.search_nodes(name=cutting_node)[0]

    for node in father.iter_descendants("postorder"):
        if node.name in Valid_:
            Valid_.pop(node.name)
        saved += 1
    # print("Saved:{}".format(saved))
    # Now, we detach this sub_tree from original
    father.detach()

    return saved

#def SavedNodesbackTrack(Graph,node,Valids):
    #for descendant in nx.descendants(Graph,node):

def Compute_Total_Saved(all_nodes, T):
    for node in all_nodes:
        saved = 1
        father = T.search_nodes(name=node)[0]
        for child in father.iter_descendants("postorder"):
            saved += 1
        all_nodes[node]['saved'] = saved

def Detach_Node_List(cutting_node,all_nodes,T):
    # Search Node in T
    father = T.search_nodes(name=cutting_node)[0]

    # Traverse all his childs
    for node in father.iter_descendants("postorder"):
        # Detach for Valid List
        if node.name in all_nodes:
            all_nodes.pop(node.name)

    # At the end Detach Node from Tree
    father.detach()
    return 0

def SolutionPath( Solution, init_pos):
    a_x = init_pos[0]
    a_y = init_pos[1]
    # print("Initial agent pos:{p}".format(p=[a_x, a_y]))
    Path = []
    action = []
    for node in Solution:
        # print("Reaching Node:{n}".format(n=node))
        # print("From Agent pos:{p}".format(p=[a_x, a_y]))
        while node != [a_x, a_y]:
            # Relative Position of Agent
            x = node[0] - a_x
            y = node[1] - a_y

            # Step to node
            if x < 0: a_x -= 1
            if x > 0: a_x += 1

            if y < 0: a_y -= 1
            if y > 0: a_y += 1

            # If we reach final node in solution, then dig it
            if node[0] == a_x and node[1] == a_y:
                dig = 1
            else:
                dig = 0

            # Action in env to construct Path
            if x < 0 and y < 0: action = [0, dig]
            if x < 0 and y > 0: action = [2, dig]
            if x > 0 and y < 0: action = [6, dig]
            if x > 0 and y > 0: action = [8, dig]
            if x == 0 and y < 0: action = [3, dig]
            if x == 0 and y > 0: action = [5, dig]
            if x > 0 and y == 0: action = [7, dig]
            if x < 0 and y == 0: action = [1, dig]

            Path.append(action)
    return Path


def generateInstance(load,path,directory):
    # Return [Tree,s_fire,Dist_Matrix,seed,scale,ax_pos,ay_pos]
    if load:
        T = nx.read_adjlist(path / directory / "MFF_Tree.adjlist")
        # Relabeling Nodes
        mapping = {}
        for node in T.nodes:
            mapping[node] = int(node)
        T = nx.relabel_nodes(T, mapping)
        T_Ad_Sym = np.load(path / directory / "FDM_MFFP.npy")
        lay = open(path / directory / "layout_MFF.json")
        pos = {}
        pos_ = json.load(lay)

        for position in pos_:
            pos[int(position)] = pos_[position]
        # Get Instance Parameters
        p = open(path / directory / "instance_info.json")
        parameters = json.load(p)
        N = parameters["N"]
        seed = parameters["seed"]
        scale = parameters["scale"]
        starting_fire = parameters["start_fire"]
        tree_height= parameters["tree_height"]

        T = nx.bfs_tree(T, starting_fire)
        T.add_node(N)

        degrees = T.degree()
        max_degree = max(j for (i, j) in degrees)
        root_degree = T.degree[starting_fire]


        return T, N, starting_fire, T_Ad_Sym, seed, scale, N, max_degree, root_degree, tree_height
