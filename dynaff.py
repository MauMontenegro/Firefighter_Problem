import argparse
import sys
from dynaff import setupExperiment, getExpConfig
from dynaff.utilities import utils
import numpy as np
from matplotlib import pyplot as plt
import gym_cellular_automata as gymca
import json

# Performance Measures
import tracemalloc
import time as tm
import os


class ExperimentLog:
    def __init__(self, path, file_name):
        self.path = path
        self.file = file_name + '.json'
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.full_path = os.path.join(self.path, self.file)

    def log_save(self, stats):
        with open(self.full_path, 'w') as fp:
            json.dump(stats, fp)


# Performance
def tracing_start():
    tracemalloc.stop()
    print("nTracing Status : ", tracemalloc.is_tracing())
    tracemalloc.start()
    print("Tracing Status : ", tracemalloc.is_tracing())


def tracing_mem():
    first_size, first_peak = tracemalloc.get_traced_memory()
    peak = first_peak / (1024 * 1024)
    print("Peak Size in MB - ", peak)
    return peak


# -------------------------------------------------------------

def argParser(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''\
        List of Available Inputs:
            -Cellular Automata Environment (caenv)
            -Random Tree (rndtree)
            ''',
        epilog='''python dynaff.py -s dpsolver_mau -i caenv -c rndtree_config'''
    )

    parser.add_argument(
        '--input', '-i', type=str,
        help="Type of Input for Solver.")
    parser.add_argument(
        '--solver', '-s', type=str,
        help="Type of Solver.")
    parser.add_argument(
        '--config', '-c', type=str,
        help="Config File.")

    return parser.parse_known_args(args)[0]


if __name__ == '__main__':
    # Get Input Arguments
    args = argParser(sys.argv[:])

    # Retrieve Experiment Configuration yaml file
    exp_config = getExpConfig(args.config)

    # Get Solver Function and Input Manager Function
    input_manager, solver = setupExperiment(args.input, args.solver)

    # Node Sizes for different experiments
    n_nodes_grid = exp_config['experiment']['Size']
    saved_p_nodes = []

    # Experiments for every Node Size
    for n_nodes in n_nodes_grid:
        # Experiment Directory and create logger for storing results
        path = str(args.input) + '/' + str(args.solver) + '/'

        file = str(exp_config['experiment']['Seed']) + '-' + \
               str(n_nodes) + '-' + \
               str(exp_config['experiment']['Initial_pos']) + '-' + \
               str(exp_config['experiment']['Env_Update'])

        # Create Logger Class
        logger = ExperimentLog(path, file)

        # Get Input for Selected Solver and config File.
        input = input_manager(exp_config, path, file, n_nodes)

        # Get Input Variables
        agent_pos_x = input[0][0]
        agent_pos_y = input[0][1]
        agent_pos = agent_pos_x, agent_pos_y
        nodes = input[1]
        Forest = input[2]
        time = input[3]
        max_budget = input[4]
        config = input[5]
        plotting = input[6]


        stats = {}
        stats['env_type'] = exp_config['experiment']['env_type']
        stats['init_pos'] = int(agent_pos_x), int(agent_pos_y)
        stats['seed'] = exp_config['experiment']['Seed']
        stats['max_budget'] = max_budget

        # Call The solver
        tracing_start()
        start = tm.time()
        print(Forest.get_ascii(show_internal=True))  # Printing Tree Structure
        max_saved_trees, Hash_Calls, Sol = solver([agent_pos_x, agent_pos_y], nodes, Forest, time, max_budget, 0,
                                                  config, 0)
        end = tm.time()
        print("time elapsed {} milli seconds".format((end - start) * 1000))
        peak = tracing_mem()

        saved_p_nodes.append(max_saved_trees)

        stats['per_time'] = (end - start) * 1000
        stats['per_mem'] = peak

        print("\nForest with:{n} nodes".format(n=len(nodes)))
        print("\nMax saved Trees:{t}".format(t=max_saved_trees))
        print("\nHash Repeated Calls:{h}".format(h=Hash_Calls))
        print("Hash Table has {l} different Forest Conditions".format(l=len(Sol)))

        # Construct Solution from Hash Table
        if args.solver == "dpsolver_mau":
            Solution = utils.Find_Solution(Forest, time, [agent_pos_x, agent_pos_y], Sol, config, plotting, exp_config['experiment']['Env_Update'] )
            print("\nSolution: {s}".format(s=Solution))
        if args.solver == "hd_heuristic" or args.solver == "ms_heuristic":
            print(config[2])
            print("\nSolution: {s}".format(s=Sol[0]))
            print("\nSolution Times: {s}".format(s=Sol[1]))
            print("\nFireline Level: {s}".format(s=Sol[2]))

        stats['sol'] = Solution
        stats['max_sav_trees'] = max_saved_trees
        stats['hash_calls'] = Hash_Calls

        logger.log_save(stats)



        if exp_config['experiment']['env_type'] == "caenv":
            ProtoEnv = gymca.prototypes[1]
            N = exp_config['experiment']['Size']
            env = ProtoEnv(nrows=N, ncols=N)
            obs = env.reset()

            # Build Path of actions that agent must follow considering DP Solution
            Sol_Path = utils.SolutionPath(Solution, agent_pos)
            print("\nPath Solution of actions:\n{p}".format(p=Sol_Path))

            total_reward = 0.0
            done = False
            step = 0
            threshold = 60

            while not done and step < threshold:
                # fig = env.render(mode="rgb_array")
                # fig.savefig('Images/Emulation_{f}.png'.format(f=step))
                # plt.close()
                # Follow Solution Path until empty.
                if len(Sol_Path) > 0:
                    action = Sol_Path[0]
                    Sol_Path.pop(0)
                else:
                    action = [4, 0]
                # print(action)
                obs, reward, done, info = env.step(action)
                total_reward += reward
                step += 1

            print(f"Total Steps: {step}")
            print(f"Total Reward: {total_reward}")

    nodes = np.array(exp_config['experiment']['Size'])
    print("Saved Trees")
    print(saved_p_nodes)
    plt.plot(nodes, saved_p_nodes)
    plt.savefig(args.solver, format="PNG")
    plt.close()
