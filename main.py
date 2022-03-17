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
    # In arbitrary_grid first node must be the fire root
    all_nodes = {}
    root = []
    c = 0
    with open('arbitrary_grid') as file:
        reader = csv.reader(file)
        for row in reader:
            label = '{x}-{y} '.format(x=row[0], y=row[1])
            if c == 0:
                root.append(label)
            all_nodes[label] = {}
            all_nodes[label]['visited'] = False
            c += 1
    return all_nodes, root[0], c


def TreeConstruct(all_nodes, t):
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
    print(all_nodes)


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


def DetachNode(t, cutting_node):
    saved = 0
    father = t.search_nodes(name=cutting_node)[0]
    for node in father.iter_descendants("postorder"):
        saved += 1
    father.detach()
    return saved


def ComputeTime(a_pos, node_pos):
    x = node_pos.split("-")
    n_x = int(x[0])
    n_y = int(x[1])
    delta_time = max(abs(n_x - a_pos[0]), abs(n_y - a_pos[1]))

    return delta_time


def Feasible(node_pos, a_pos, time, level, max_budget):
    # Compute Ticks to reach node_pos from root
    t_node = level * 2
    # Compute elapsed ticks
    e_time = max_budget - time
    # Compute ticks from agent to node_pos
    d_time = ComputeTime(a_pos, node_pos)
    # Elapsed time + time to reach node
    t_time = e_time + d_time

    # print("t_node:{t}".format(t=t_node))
    # print("t_time:{t}".format(t=t_time))

    # If agent can reach node before fire gets him (elapsed time plus time to reach node mus be less than level ticks)
    if t_node > t_time:
        return True
    else:
        return False


def DPSolution(a_pos, nodes, F, time, Hash, max_budget):
    # F is the actual Forest
    # a_pos is the actual agent position
    # Remaining time
    # node list in the Forest
    # ------------------------------------------
    # print("Recursion")
    # print("Agent:{a} , nodes:{n} , Forest:{f}, time:{t}".format(a=a_pos, n=nodes, f=F.get_ascii(show_internal=True),
    # t=time))
    # Base Conditions(Tree is empty or time is over)
    if F.is_leaf() == True or time == 0:
        return 0

    # Construct the Key for Hash ( String: "Forest;time;pos_x,pos_y" )
    key = F.write(format=8) + ';' + str(time) + ';' + str(a_pos[0]) + ',' + str(a_pos[1])

    # Search if we already see this Forest Conditions
    if key in Hash:
        #print("Forest Already Seen")
        return Hash[key]['value']
    else:
        Hash[key] = {}

    # Compute Feasible Nodes in actual Forest
    #print("Position:{pos}".format(pos=a_pos))
    Valid = {}
    for node in nodes:
        if Feasible(node, a_pos, time, nodes[node]['level'], max_budget) == True:
            Valid[node] = {}
            Valid[node]['level'] = nodes[node]['level']

    #print(Valid)

    # Travel valid nodes and play the recursion
    saved = 0
    for valid_node in Valid:
        # Copy of Tree and Valid to send to each node prunning
        #print("Evaluating valid node:{n}".format(n=valid_node))
        F_copy = F.copy("newick")
        Valid_copy = Valid.copy()
        Valid_copy.pop(valid_node)  # This node will be pruned, so next iter will no ve valid

        # Computes Saved Nodes and prune Tree, also modify the next list of valid nodes (deleting pruned nodes)
        saved = SavedNodes(F_copy, valid_node, Valid_copy)

        #print("New Valid")
        #print(Valid_copy)

        # Computes Remaining Time if agent travel to this Valid node
        t_ = time - ComputeTime(a_pos, valid_node)

        # New agent position moves to node position
        x = valid_node.split("-")
        n_x = int(x[0])
        n_y = int(x[1])

        value = DPSolution([n_x, n_y], Valid_copy, F_copy, t_, Hash, max_budget)
        Valid[valid_node]['value'] = value + saved
        #print("Valid node values:")
        #print(Valid)
    #print("Leaving Recursion")
    if Valid:
        max_value = max(int(d['value']) for d in Valid.values())
        max_key_node = max(Valid, key=lambda v: Valid[v]['value'])
        Hash[key]['max_node'] = max_key_node
        Hash[key]['value'] = max_value
        return max_value
    else:
        Hash[key]['value'] = saved
        return saved


def Find_Solution(Forest, Time, a_pos, Hash):
    # Construct the Key for Hash ( String: "Forest;time;pos_x,pos_y" )
    key = Forest.write(format=8) + ';' + str(Time) + ';' + str(a_pos[0]) + ',' + str(a_pos[1])
    Solution = []
    while Hash[key]['value'] != 0:
        # Getting node of max value
        node = Hash[key]['max_node']
        #print("Node:{n}".format(n=node))
        # Saved Trees by selecting this node
        saved = DetachNode(Forest, node)
        #print("Value of Node:{v}".format(v=saved))
        # Computes Remaining Time if agent travel to this node
        Time = Time - ComputeTime(a_pos, node)
        # New agent position moves to node position
        x = node.split("-")
        a_pos[0] = int(x[0])
        a_pos[1] = int(x[1])
        # Append Node to Solution
        Solution.append([a_pos[0], a_pos[1]])
        # New Key
        key = Forest.write(format=8) + ';' + str(Time) + ';' + str(a_pos[0]) + ',' + str(a_pos[1])
        # print(Forest.get_ascii(show_internal=True))

    return Solution


def SolutionPath(Solution, init_pos):
    a_x = init_pos[0]
    a_y = init_pos[1]
    #print("Initial agent pos:{p}".format(p=[a_x, a_y]))
    Path = []
    action = []
    for node in Solution:
        #print("Reaching Node:{n}".format(n=node))
        #print("From Agent pos:{p}".format(p=[a_x, a_y]))
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


if __name__ == '__main__':
    # Environment Variables
    grid = obs[0]
    agent_pos_x, agent_pos_y = obs[1][1][0], obs[1][1][1]
    threshold = 30

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
    TreeConstruct(all_nodes, F)

    # 5. Checking and Saving
    # t.write(format=1, outfile="burning_tree.nw")  # Saving Original Tree
    print(F.get_ascii(show_internal=True))  # Printing Tree Structure

    # Variables
    tree_levels = max(int(d['level']) for d in all_nodes.values())
    env_update = 4  # Update ratio of environment respect to agent (2:1)
    time = tree_levels * env_update  # Budget Time before Tree burns entirely
    max_budget = tree_levels * env_update
    Hash = {}  # Here, we store a Forest with specific conditions and his max value of saved trees

    # Dynamic Programming Algorithm
    all_nodes.pop(root_label)
    # OPT(Initial Agent Position, Full Forest,Max Budget Time)
    max_saved_trees = DPSolution([agent_pos_x, agent_pos_y], all_nodes, F, time, Hash, max_budget)

    print("\n\nMax saved Trees:{t}".format(t=max_saved_trees))
    print("Hash Table has {l} different Forest Conditions".format(l=len(Hash)))

    # Construct Solution from Hash Table
    Solution = Find_Solution(F, time, [agent_pos_x, agent_pos_y], Hash)
    print("\n\nSolution: {s}".format(s=Solution))

    # Build Path of actions that agent must follow considering DP Solution
    Sol_Path = SolutionPath(Solution, [agent_pos_x, agent_pos_y])
    print("Path Solution of actions:\n{p}".format(p=Sol_Path))

    # Fire Propagation Test
    while not done and step < threshold:
        fig = env.render(mode="rgb_array")
        fig.savefig('Images/Emulation_{f}.png'.format(f=step))
        plt.close()
        # Follow Solution Path until empty.
        if len(Sol_Path) > 0:
            action = Sol_Path[0]
            Sol_Path.pop(0)
        else:
            action = [4, 0]
        #print(action)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step += 1

    print(f"{ProtoEnv}")
    print(f"Total Steps: {step}")
    print(f"Total Reward: {total_reward}")
