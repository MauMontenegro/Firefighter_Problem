from dynaff.utilities import utils
from tqdm import tqdm


def Feasible(node_pos, a_pos, time, level, max_budget, config):
    # Compute Ticks to reach node_pos from root
    t_node = level * 4
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


def dpsolver_mau(a_pos, nodes, F, time, max_budget, hash_calls, config, recursion):
    # F is the actual Forest
    # a_pos is the actual agent position
    # Remaining time
    # node list in the Forest
    # hash_calls track valid hash calls
    # ------------------------------------------

    h = hash_calls

    # Base Conditions(Tree is empty or time is over)
    if F.is_leaf() == True or time == 0:
        return 0, h

    # Construct the Key for Hash ( String: "Forest;time;pos_x,pos_y" )
    key = F.write(format=8) + ';' + str(time) + ';' + str(a_pos[0]) + ',' + str(a_pos[1])

    # Search if we already see this Forest Conditions. If not, create new key entry
    if key in Hash:
        h += 1
        return Hash[key]['value'], h
    else:
        Hash[key] = {}

    # Compute Feasible Nodes in actual Forest and add to Valid Node Dictionary
    Valid = {}
    for node in nodes:
        if Feasible(node, a_pos, time, nodes[node]['level'], max_budget, config):
            Valid[node] = {}
            Valid[node]['level'] = nodes[node]['level']

    saved = 0
    # Progress Bar Creation
    if time == max_budget:
        pbar = tqdm(total=len(Valid))

    # Traverse Valid node list and compute his value by recurrence
    for valid_node in Valid:
        if time == max_budget:
            pbar.update(1)

        # Copy of Tree and Valid list to send in following recurrence
        F_copy = F.copy("newick")

        # In metric distance envs like "caenv", only send to the following recurrence a copy of valid
        # cause once a node is invalid due delta_time from agent position it never will be valid again.
        # When there are no metric distance envs, like "rndtree", we send all nodes in actual forest (except pruned)
        if config[1] == 1:
            Valid_copy = Valid.copy()
        if config[1] == 0 :
            Valid_copy = nodes.copy()

        # This node will be pruned, so next iter will not ve valid(metric or no metric distances)
        Valid_copy.pop(valid_node)

        # Compute Saved Nodes and prune Tree, also modify the next list of valid nodes (deleting pruned nodes)
        saved = utils.SavedNodes(F_copy, valid_node, Valid_copy)

        # Compute Remaining Time if agent travel to this Valid node
        t_ = time - utils.ComputeTime(a_pos, valid_node, config)

        # New agent position moves to node position
        if config[0] == "caenv":
            x = valid_node.split("-")
            n_x = int(x[0])
            n_y = int(x[1])
        if config[0] == "rndtree":
            n_x = valid_node
            n_y = valid_node

        # Solve next sub-problem
        value, h = dpsolver_mau([n_x, n_y], Valid_copy, F_copy, t_, max_budget, h, config, recursion +1)

        # Fill Valid Node space by is returning best value plus is current saved trees
        Valid[valid_node]['value'] = value + saved

    # Once all valid nodes values are computed then calculate the best (max) and store in current key
    # along with the node or position that belongs to that value
    if Valid:
        max_value = max(int(d['value']) for d in Valid.values())
        max_key_node = max(Valid, key=lambda v: Valid[v]['value'])
        Hash[key]['max_node'] = max_key_node
        Hash[key]['value'] = max_value
        if recursion == 0:
            return max_value, h, Hash
        return max_value, h
    # If there are no valid nodes for current key then only return saved trees
    else:
        Hash[key]['value'] = saved
        if recursion == 0:
            return saved, h, Hash
        return saved, h
