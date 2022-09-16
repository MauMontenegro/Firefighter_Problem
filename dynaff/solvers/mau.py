from dynaff.utilities import utils
from tqdm import tqdm


def Feasible(node_pos, a_pos, time, level, max_budget, config):
    """Function that decides if node is Feasible by comparing agent travel time to a node and fire travel time
     to same node

    :param node_pos: Int Node name
    :param a_pos: Real agent position
    :param time: Real Remaining time
    :param level: Int Node level
    :param max_budget: Real Max initial budget time
    :param config:
    :return: Bool
        True if node is feasible and False if not
    """

    # Compute Ticks to reach node_pos from root
    t_node = level * 1
    # Compute elapsed ticks
    e_time = max_budget - time
    # Compute ticks from agent to node_pos
    d_time = utils.ComputeTime(a_pos, node_pos, config)
    # Elapsed time + time to reach node
    t_time = e_time + d_time

    # If agent can reach node before fire gets him (elapsed time plus time to reach node mus be less than level ticks)
    if t_node > t_time:
        return True
    else:
        return False


# Global Hash and Grid
Hash = {}


def dpsolver_mau(a_pos, nodes, F, time, max_budget, hash_calls, dist_matrix, recursion):
    """ Wise-Recursive Dynamic Programming Solver for Moving Firefighter Problem

    Recursive Dynamic Programming function. It takes a Forest in ete3 form and construct a key with the state of the
    Forest (remaining_time, agent position and remaining nodes that are not already burning or defended) and store in
    a Hash table. Function computes feasible nodes due current state and for each feasible node call the recurrence
    computing saved nodes until arrive a 'Base Case' and start to go back with Values. At the end we will have a Hash
    Table with state as key along with his optimal 'next_node-value'. Wise part is for only store in Hash states that
    currently happens, cause due to fire dynamics some states will never arise. So, memoization will be most
    time-consuming with a little benefit.

    :param a_pos: Real vector containing x_position and y_position for the agent
    :param nodes: Dictionary containing all nodes with degree and level
    :param F: Forest in ete3 form
    :param time: Real Remaining time until forest consumption
    :param max_budget: Real Initial max-budget time
    :param hash_calls: Int Number of times Hash Table has the key we are evaluating
    :param config: Additional information for different inputs (like Adjacency matrix)
    :param recursion: Number of times recursion is called
    :return:
        Saved Nodes for state evaluation
        Actual recursion
        Hash Table
    """

    h = hash_calls
    # Base Conditions of recursion (Tree is empty or time is over)
    if (F.is_leaf() == True) or (time <= 0):
        return 0, h

    # Construct the Key to store in Hash
    # String: " Forest_newick; remaining_time; a_pos_x, a_pos_y "
    key = F.write(format=8) + ';' + str(time) + ';' + str(a_pos)

    # Search if we already see this Forest Conditions. If not, create new key entry
    if key in Hash:
        h += 1
        return Hash[key]['value'], h
    else:
        Hash[key] = {}

    # Compute Feasible Nodes in actual Forest and add to Valid Node Dictionary
    Valid = {}
    for node in nodes:
        if Feasible(node, a_pos, time, nodes[node]['level'], max_budget, dist_matrix):
            Valid[node] = {}
            Valid[node]['level'] = nodes[node]['level']

    saved = 0
    pbar = 0

    # Progress Bar Creation
    if recursion == 0:
        pbar = tqdm(total=len(Valid))

    # Traverse Valid node list and compute his value by recurrence
    for valid_node in Valid:
        if recursion == 0:
            pbar.update(1)

        # Copy of Tree and Valid list to send in following recurrence
        F_copy = F.copy("newick")

        # In metric distance envs like "caenv", only send to the following recurrence a copy of 'Valid'
        # cause once a node is invalid due delta_time from agent position it never will be valid again.
        # When there are no metric distance envs, like "rndtree", we send all nodes in actual forest (except pruned)
        Valid_copy = Valid.copy()

        # This node will be pruned, so next iter will not ve valid(metric or no metric distances)
        Valid_copy.pop(valid_node)

        # Compute Saved Nodes and prune Tree, also modify the next list of valid nodes (deleting pruned nodes)
        saved = utils.SavedNodes(F_copy, valid_node, Valid_copy)

        # Compute Remaining Time if agent travel to this Valid node
        t_ = time - utils.ComputeTime(a_pos, valid_node, dist_matrix)

        # New agent position moves to node position
        n_x = valid_node

        # Solve next sub-problem
        value, h = dpsolver_mau(n_x, Valid_copy, F_copy, t_, max_budget, h, dist_matrix, recursion + 1)

        # Assign Valid Node value by his returning best value + is current saved trees
        Valid[valid_node]['value'] = value + saved

    # Once all valid nodes values are computed then calculate the best (max) and store in current key
    # along with the node or position that belongs to that value
    if Valid:
        max_value = max(int(d['value']) for d in Valid.values())
        max_key_node = max(Valid, key=lambda v: Valid[v]['value'])
        Hash[key]['max_node'] = max_key_node
        Hash[key]['value'] = max_value
        if recursion == 0:             # We already do all the recursions
            return max_value, h, Hash
        return max_value, h
    # If there are no valid nodes for current key then only return saved trees
    else:
        Hash[key]['value'] = saved
        if recursion == 0:
            return saved, h, Hash
        return saved, h


def hd_heuristic(a_pos, nodes, F, time, max_budget, hash_calls, config, recursion):
    """Heuristic that saves feasible nodes with maximum degree

    :param a_pos:
    :param nodes:
    :param F:
    :param time:
    :param max_budget:
    :param hash_calls:
    :param config:
    :param recursion:
    :return:
    """
    # Control Variables
    saved = 0
    solution = []
    Valid = {}
    len_valid = 1
    fireline = []
    time_travel = []

    # Loop while there are available nodes to evaluate
    while len_valid > 0:
        Valid.clear()
        # Compute Feasible Nodes
        for node in nodes:
            if Feasible(node, a_pos, time, nodes[node]['level'], max_budget, config):
                Valid[node] = {}
                Valid[node]['level'] = nodes[node]['level']
                Valid[node]['degree'] = nodes[node]['degree']

        len_valid = len(Valid)
        if len_valid > 0:
            # For valid Nodes Get Max Degree Value with his Key
            max_degree = max(int(d['degree']) for d in Valid.values())
            max_degree_node = max(Valid, key=lambda v: Valid[v]['degree'])
            solution.append(max_degree_node)

            # Remove node from actual Valid List
            Valid.pop(max_degree_node)

            # Compute Saved Nodes and detach max degree node
            saved += utils.SavedNodes(F, max_degree_node, Valid)

            # New Nodes on next iteration will Be Valid This in this
            nodes.clear()
            nodes = Valid.copy()

            # Compute Remaining Time if agent travel to this Valid node
            # Compute Remaining Time if agent travel to this Valid node
            elapsed_time = utils.ComputeTime(a_pos, max_degree_node, config)
            time -= elapsed_time
            fireline_level = (max_budget - time) / 2
            fireline.append(fireline_level)
            time_travel.append(elapsed_time)

            # New agent position moves to node position
            if config[0] == "caenv":
                x = max_degree_node.split("-")
                a_pos[0] = int(x[0])
                a_pos[1] = int(x[1])
            if config[0] == "rndtree":
                a_pos = max_degree_node

    return saved, 0, [solution, time_travel, fireline]


def ms_heuristic(a_pos, nodes, F, time, max_budget, hash_calls, config, recursion):
    """Heuristic that saves nodes with maximum children

    :param a_pos:
    :param nodes:
    :param F:
    :param time:
    :param max_budget:
    :param hash_calls:
    :param config:
    :param recursion:
    :return:
    """
    # Control Variables
    saved = 0
    solution = []
    time_travel = []
    fireline = []
    Valid = {}
    len_valid = 1

    # First, Compute Total Saved Trees for each node
    utils.Compute_Total_Saved(nodes, F)
    print(nodes)
    # Loop while there are available nodes to evaluate
    while len_valid > 0:
        Valid.clear()
        # Compute Feasible Nodes
        for node in nodes:
            if Feasible(node, a_pos, time, nodes[node]['level'], max_budget, config):
                Valid[node] = {}
                Valid[node]['level'] = nodes[node]['level']
                Valid[node]['degree'] = nodes[node]['degree']
                Valid[node]['saved'] = nodes[node]['saved']
        print(Valid)
        len_valid = len(Valid)
        if len_valid > 0:
            # For valid Nodes Get Max Saved Values with his Key
            max_saved = max(int(d['saved']) for d in Valid.values())
            max_saved_node = max(Valid, key=lambda v: Valid[v]['saved'])

            solution.append(max_saved_node)

            # Remove node from actual Valid List
            Valid.pop(max_saved_node)

            # Compute Saved Nodes and detach max degree node
            saved += max_saved
            utils.Detach_Node_List(max_saved_node, Valid, F)

            # New Nodes on next iteration will Be Valid This in this
            nodes.clear()
            nodes = Valid.copy()

            # Update Max Saved Nodes List
            utils.Compute_Total_Saved(nodes, F)

            # Compute Remaining Time if agent travel to this Valid node
            elapsed_time = utils.ComputeTime(a_pos, max_saved_node, config)
            time -= elapsed_time
            fireline_level = (max_budget - time) / 2
            fireline.append(fireline_level)
            time_travel.append(elapsed_time)

            # New agent position moves to node position
            if config[0] == "caenv":
                x = max_saved_node.split("-")
                a_pos[0] = int(x[0])
                a_pos[1] = int(x[1])
            if config[0] == "rndtree":
                a_pos = max_saved_node

    return saved, 0, [solution, time_travel, fireline]
