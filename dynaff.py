import argparse
import sys
import numpy as np
from matplotlib import pyplot as plt
import gym_cellular_automata as gymca
import json

# Performance Measures
import tracemalloc
import time as tm
import os

from dynaff import setupExperiment, getExpConfig
from dynaff.utilities import utils


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
    # Get input arguments for input_instance_type, solver and config_file
    args = argParser(sys.argv[:])

    # Retrieve Experiment Configuration yaml file
    exp_config = getExpConfig(args.config)

    # Get Solver Function and Input Manager Function
    input_manager, solver = setupExperiment(args.input, args.solver)

    # Graph Size for different experiments
    n_nodes_grid = exp_config['experiment']['Size']
    saved_p_nodes = []

    # Experiments for every Graph
    for n_nodes in n_nodes_grid:
        # Experiment Directory and create logger for store results
        path = str(args.input) + '/' + str(args.solver) + '/'
        file = str(exp_config['experiment']['Seed']) + '-' + \
               str(n_nodes) + '-' + \
               str(exp_config['experiment']['Env_Update'])

        logger = ExperimentLog(path, file)

        # Get Instance for Selected Solver and config File.
        input = input_manager(exp_config, path, file, n_nodes)
        agent_pos_x = input[0][0]
        agent_pos_y = input[0][1]
        agent_pos = agent_pos_x, agent_pos_y
        nodes = input[1]
        Forest = input[2]
        time = input[3]
        max_budget = input[4]
        config = input[5]
        plotting = input[6]

        # Call The solver
        # ----------------------------------------------------------------------------------------------------------
        tracing_start()
        start = tm.time()
        max_saved_trees, Hash_Calls, Sol = solver([agent_pos_x, agent_pos_y], nodes, Forest, time, max_budget, 0,
                                                  config, 0)
        end = tm.time()
        print("time elapsed {} milli seconds".format((end - start) * 1000))
        peak = tracing_mem()
        # -----------------------------------------------------------------------------------------------------------

        # Saved nodes per Graph
        saved_p_nodes.append(max_saved_trees)

        # Just Printing Results
        print("\nForest with:{n} nodes".format(n=len(nodes)))
        print("\nMax saved Trees:{t}".format(t=max_saved_trees))
        print("\nHash Repeated Calls:{h}".format(h=Hash_Calls))
        print("Hash Table has {l} different Forest Conditions".format(l=len(Sol)))

        # Retrieve Solution Strategy
        if args.solver == "dpsolver_mau":
            Sol, Solution_times, Solution_elapsed = utils.Find_Solution(Forest, time, [agent_pos_x, agent_pos_y],
                                                                             Sol, config, plotting,
                                                                             exp_config['experiment']['Env_Update'])
            print("\nSolution: {s}".format(s=Sol))
            print("Time elapsed by step: {s}".format(s=Solution_times))
            print("Total Elapsed time by step: {s}".format(s=Solution_elapsed))
        if args.solver == "hd_heuristic" or args.solver == "ms_heuristic":
            print("\nSolution: {s}".format(s=Sol[0]))
            print("\nTime elapsed by step: {s}".format(s=Sol[1]))
            print("\nFireline Level: {s}".format(s=Sol[2]))
            utils.graphSol(Sol, plotting)

        # Saving stats for general parameters
        stats = {}
        stats['env_type'] = exp_config['experiment']['env_type']
        stats['init_pos'] = int(agent_pos_x), int(agent_pos_y)
        stats['seed'] = exp_config['experiment']['Seed']
        stats['max_budget'] = max_budget

        # Saving performance stats
        stats['per_time'] = (end - start) * 1000
        stats['per_mem'] = peak

        # Saving solution stats
        stats['sol'] = Sol
        stats['max_sav_trees'] = max_saved_trees
        stats['hash_calls'] = Hash_Calls

        logger.log_save(stats)

        if exp_config['experiment']['env_type'] == "caenv":
            ProtoEnv = gymca.prototypes[1]
            N = exp_config['experiment']['Size']
            env = ProtoEnv(nrows=N, ncols=N)
            obs = env.reset()

            # Build Path of actions that agent must follow considering DP Solution
            Sol_Path = utils.SolutionPath(Sol, agent_pos)
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
    print("Saved Trees: {s}".format(s=saved_p_nodes))
    plt.plot(nodes, saved_p_nodes)
    plt.savefig(args.solver, format="PNG")
    plt.close()
