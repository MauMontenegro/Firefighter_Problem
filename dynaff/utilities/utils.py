import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import os


def ComputeTime(a_pos, node_pos, config):
    delta_time = 0
    if config[0] == "caenv":
        x = node_pos.split("-")
        n_x = int(x[0])
        n_y = int(x[1])
        delta_time = max(abs(n_x - a_pos[0]), abs(n_y - a_pos[1]))
    if config[0] == "rndtree":
        Adj = config[2]
        delta_time = Adj[int(node_pos)][int(a_pos[0])]
    return delta_time


def DetachNode(t, cutting_node, levels, saved_nodes):
    saved = 1
    father = t.search_nodes(name=cutting_node)[0]
    for node in father.iter_descendants("postorder"):
        saved += 1
        levels.pop(int(node.name))
        saved_nodes.append(int(node.name))
    father.detach()
    return saved


def Find_Solution(Forest, Time, a_pos, Hash, config, plotting, update):
    # Construct the Key for Hash ( String: "Forest;time;pos_x,pos_y" )
    frame = 0
    total_budget = Time
    spread_time = update
    last_level_burnt = 0
    last_level_burnt_ = 0
    elapsed = 0
    labels = {}
    # Labeling Nodes
    for node in plotting[0]:
        labels[int(node)] = str(node)

    key = Forest.write(format=8) + ';' + str(Time) + ';' + str(a_pos[0]) + ',' + str(a_pos[1])
    graphSolution(plotting, frame, labels)
    Solution = []
    Solution_times = []
    Solution_elapsed_time = []

    labels.pop(plotting[4])
    while Hash[key]['value'] != 0:
        frame += 1
        # Getting node of max value
        node = Hash[key]['max_node']
        # Computes Remaining Time if agent travel to this node
        time_ = ComputeTime(a_pos, node, config)
        Time = Time - time_
        elapsed += (total_budget - Time)
        Solution_elapsed_time.append(elapsed)

        print("Time to travel")
        print(time_)
        print("Levels that already burned")
        print((total_budget - Time) / spread_time + last_level_burnt_)

        all_levels = plotting[6]
        levels_to_burnt_ = (total_budget - Time) / spread_time + (last_level_burnt_ - int(last_level_burnt_))
        levels_to_burnt = int(levels_to_burnt_)

        # Burn nodes in Tree
        for level in range(last_level_burnt + 1, (last_level_burnt + levels_to_burnt) + 1):
            keys = [k for k, v in all_levels.items() if v <= level]
            # Assign to burning nodes and quit from remaining
            for element in keys:
                if element not in plotting[2]:
                    plotting[2].append(element)
                if element in plotting[3]:
                    plotting[3].remove(element)
            graphSolution(plotting, frame, labels, 0, level, spread_time * level)
            frame += 1

        # Change pos of agent to next node in solution
        pos = plotting[1]

        pos[plotting[4]] = pos[int(node)]
        plotting[1] = pos
        # Saved Trees by selecting this node
        saved = DetachNode(Forest, node, plotting[6], plotting[7])

        last_level_burnt_ += levels_to_burnt_
        last_level_burnt = int(last_level_burnt_)
        # Add saved Node
        plotting[7].append(int(node))

        # New agent position moves to node position
        if config[0] == "caenv":
            x = node.split("-")
            a_pos[0] = int(x[0])
            a_pos[1] = int(x[1])
            Solution.append([a_pos[0], a_pos[1]])
        if config[0] == "rndtree":
            a_pos[0] = node
            a_pos[1] = node
            Solution.append(node)
            Solution_times.append(time_)
        # New Key
        graphSolution(plotting, frame, labels, total_budget - Time, levels_to_burnt, elapsed)
        total_budget = Time
        key = Forest.write(format=8) + ';' + str(Time) + ';' + str(a_pos[0]) + ',' + str(a_pos[1])

    return Solution, Solution_times, Solution_elapsed_time


def graphSolution(plot_config, frame_num, labels, spend_time=0, burned_levels=0, total_elapsed=0):
    Tree = plot_config[0]
    pos = plot_config[1]
    burnt_nodes = plot_config[2]
    remaining_nodes = plot_config[3]
    agent_pos = plot_config[4]
    path = plot_config[5]
    saved_nodes = plot_config[7]

    figure(figsize=(9, 9), dpi=80)

    # This is for get max and min labels in plotting
    max_x_value = max(d[0] for d in pos.values())
    min_x_value = min(d[0] for d in pos.values())
    max_y_value = max(d[1] for d in pos.values())
    min_y_value = min(d[1] for d in pos.values())

    # Nodes
    options = {"edgecolors": "tab:gray", "node_size": 550, "alpha": 1}
    options_burned = {"edgecolors": "tab:gray", "node_size": 550, "alpha": 0.7}
    nx.draw_networkx_nodes(Tree, pos, nodelist=burnt_nodes, node_color="tab:red", label="Burning", **options_burned)
    nx.draw_networkx_nodes(Tree, pos, nodelist=remaining_nodes, node_color="tab:green", label="Unlabeled", **options)
    nx.draw_networkx_nodes(Tree, pos, nodelist=saved_nodes, node_color="tab:blue", label="Saved", **options)
    nx.draw_networkx_nodes(Tree, pos, nodelist=[agent_pos], node_color="tab:gray", label="Agent", **options)
    nx.draw_networkx_labels(Tree, pos, labels, font_size=16, font_color='black')

    # Edges
    nx.draw_networkx_edges(Tree, pos, width=2.0, alpha=0.3)

    # Text, Labels and tittle

    font_title = {'family': 'abel',
                  'color': 'darkred',
                  'weight': 'normal',
                  'size': 16,
                  }

    font_axis = {'family': 'Merriweather',
                 'color': 'darkred',
                 'weight': 'normal',
                 'size': 15,
                 }

    font_txt = {'family': 'Merriweather',
                'color': 'black',
                'weight': 'normal',
                'size': 12,
                }

    plt.xticks(np.arange(min_x_value, max_x_value, 1))
    plt.yticks(np.arange(min_y_value, max_y_value, 1))
    plt.title('Moving Firefighter Problem: Tree Instance', fontdict=font_title, fontsize=20)
    plt.xlabel('X-Position', fontdict=font_axis, labelpad=20)
    plt.ylabel('Y-Position', fontdict=font_axis, labelpad=20)
    plt.legend(loc="lower left", ncol=4)

    down, left = -.30, -.30

    plt.text(min_x_value + down, min_y_value + left, "Travel time: {0:.2f}".format(spend_time),
             horizontalalignment="left",
             verticalalignment="bottom", fontdict=font_txt)
    plt.text(min_x_value + down, min_y_value + 1.2 * left, "Burned Levels: {l}".format(l=burned_levels),
             horizontalalignment="left",
             verticalalignment="bottom", fontdict=font_txt)
    plt.text(min_x_value + down, min_y_value + 1.4 * left, "Total elapsed Time: {0:.2f}".format(total_elapsed),
             horizontalalignment="left",
             verticalalignment="bottom", fontdict=font_txt)

    graph = plt.gcf()
    frame = str(frame_num)
    full_path = path + "/"
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    graph.savefig(full_path + 'frame_' + frame, format="PNG")
    plt.close()


def graphSol(Sol, plot_config):
    Tree = plot_config[0]
    pos = plot_config[1]
    burnt_nodes = plot_config[2]
    remaining_nodes = plot_config[3]
    agent_pos = plot_config[4]
    path = plot_config[5]
    saved_nodes = plot_config[7]
    solution = Sol[0]
    time_step = Sol[1]
    e_time = Sol[2]


    # This is for get max and min labels in plotting
    max_x_value = max(d[0] for d in pos.values())
    min_x_value = min(d[0] for d in pos.values())
    max_y_value = max(d[1] for d in pos.values())
    min_y_value = min(d[1] for d in pos.values())

    # Node Drawing Options
    options = {"edgecolors": "tab:gray", "node_size": 550, "alpha": 1}
    options_burned = {"edgecolors": "tab:gray", "node_size": 550, "alpha": 0.7}

    all_levels = plot_config[6]

    solution.insert(0, len(Tree)-1)
    time_step.insert(0, 0)
    e_time.insert(0, 0)

    labels = {}
    # Labeling Nodes
    for node in Tree:
        labels[int(node)] = str(node)
    frame=0

    for sol_node, elapsed, step in zip(solution, e_time, time_step):

        # Nodes that has level below elapsed time are burned
        keys = [k for k, v in all_levels.items() if v <= int(elapsed)]
        for element in keys:
            if element not in burnt_nodes:
                burnt_nodes.append(element)
            if element in remaining_nodes:
                remaining_nodes.remove(element)

        # Append defended node to and eliminate from unlabeled
        saved_nodes.append(int(sol_node))

        # Append agent actual position
        agent_pos = int(sol_node)

        figure(figsize=(9, 9), dpi=80)
        nx.draw_networkx_nodes(Tree, pos, nodelist=burnt_nodes, node_color="tab:red", label="Burning", **options_burned)
        nx.draw_networkx_nodes(Tree, pos, nodelist=remaining_nodes, node_color="tab:green", label="Unlabeled", **options)
        nx.draw_networkx_nodes(Tree, pos, nodelist=saved_nodes, node_color="tab:blue", label="Saved", **options)
        nx.draw_networkx_nodes(Tree, pos, nodelist=[agent_pos], node_color="tab:gray", label="Agent", **options)
        nx.draw_networkx_labels(Tree, pos, labels, font_size=16, font_color='black')

        # Edges
        nx.draw_networkx_edges(Tree, pos, width=2.0, alpha=0.3)

        # Text, Labels and tittle

        font_title = {'family': 'fantasy',
                      'color': 'darkred',
                      'weight': 'normal',
                      'size': 16,
                      }

        font_axis = {'family': 'fantasy',
                     'color': 'darkred',
                     'weight': 'normal',
                     'size': 15,
                     }

        font_txt = {'family': 'fantasy',
                    'color': 'black',
                    'weight': 'normal',
                    'size': 12,
                    }

        plt.xticks(np.arange(min_x_value, max_x_value, 1))
        plt.yticks(np.arange(min_y_value, max_y_value, 1))
        plt.title('Moving Firefighter Problem: Tree Instance', fontdict=font_title, fontsize=20)
        plt.xlabel('X-Position', fontdict=font_axis, labelpad=20)
        plt.ylabel('Y-Position', fontdict=font_axis, labelpad=20)
        plt.legend(loc="lower left", ncol=4)

        down, left = -.30, -.30

        plt.text(min_x_value + down, min_y_value + left, "Travel time: {0:.2f}".format(step),
                 horizontalalignment="left",
                 verticalalignment="bottom", fontdict=font_txt)
        plt.text(min_x_value + down, min_y_value + 1.2 * left, "Burned Levels: {l}".format(l=step),
                 horizontalalignment="left",
                 verticalalignment="bottom", fontdict=font_txt)
        plt.text(min_x_value + down, min_y_value + 1.4 * left, "Total elapsed Time: {0:.2f}".format(elapsed),
                 horizontalalignment="left",
                 verticalalignment="bottom", fontdict=font_txt)

        graph = plt.gcf()
        frame_ = str(frame)
        frame += 1
        full_path = path + "/"
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        graph.savefig(full_path + 'frame_' + frame_, format="PNG")
        plt.close()


def SavedNodes(t, cutting_node, Valid_):
    # First we get the corresponding branch of T
    saved = 1  # As we "defend nodes" detach node is also saved
    father = t.search_nodes(name=cutting_node)[0]

    for node in father.iter_descendants("postorder"):
        if node.name in Valid_:
            Valid_.pop(node.name)
        saved += 1
    # print("Saved:{}".format(saved))
    # Now, we detach this sub_tree from original
    father.detach()

    return saved


def Compute_Total_Saved(all_nodes, T):
    for node in all_nodes:
        saved = 1
        father = T.search_nodes(name=node)[0]
        for child in father.iter_descendants("postorder"):
            saved += 1
        all_nodes[node]['saved'] = saved


def Detach_Node_List(cutting_node, all_nodes, T):
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


def SolutionPath(Solution, init_pos):
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
