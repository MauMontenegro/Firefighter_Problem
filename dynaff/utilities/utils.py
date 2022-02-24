def ComputeTime(a_pos, node_pos, config):
    delta_time = 0
    if config[0] == "caenv":
        x = node_pos.split("-")
        n_x = int(x[0])
        n_y = int(x[1])
        delta_time = max(abs(n_x - a_pos[0]), abs(n_y - a_pos[1]))
    if config[0] == "rndtree":
        Adj = config[1]
        delta_time = Adj[int(node_pos)][int(a_pos[0])]
    return delta_time

def DetachNode(t, cutting_node):
    saved = 0
    father = t.search_nodes(name=cutting_node)[0]
    for node in father.iter_descendants("postorder"):
        saved += 1
    father.detach()
    return saved

def Find_Solution(Forest, Time, a_pos, Hash,config):
    # Construct the Key for Hash ( String: "Forest;time;pos_x,pos_y" )
    key = Forest.write(format=8) + ';' + str(Time) + ';' + str(a_pos[0]) + ',' + str(a_pos[1])
    Solution = []
    while Hash[key]['value'] != 0:
        # Getting node of max value
        node = Hash[key]['max_node']
        #print("Node:{n}".format(n=node))
        # Saved Trees by selecting this node
        saved = DetachNode(Forest, node)
        print("Value of Node:{v}".format(v=saved))
        # Computes Remaining Time if agent travel to this node
        Time = Time - ComputeTime(a_pos, node,config)
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
        # New Key
        key = Forest.write(format=8) + ';' + str(Time) + ';' + str(a_pos[0]) + ',' + str(a_pos[1])
        # print(Forest.get_ascii(show_internal=True))

    return Solution


def SavedNodes(t, cutting_node, Valid_):
    # First we get the corresponding branch of T
    saved = 0
    father = t.search_nodes(name=cutting_node)[0]

    for node in father.iter_descendants("postorder"):
        if node.name in Valid_:
            Valid_.pop(node.name)
        saved += 1
    # print("Saved:{}".format(saved))
    # Now, we detach this sub_tree from original
    father.detach()

    return saved


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