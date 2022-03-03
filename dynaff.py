import argparse
import sys
from dynaff import setupExperiment, getExpConfig
from dynaff.utilities import utils
import numpy as np
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


class Initial_pos:
    pass


if __name__ == '__main__':
    # Get Arguments
    args = argParser(sys.argv[:])

    # Retrieve Experiment Configuration yaml file
    exp_config = getExpConfig(args.config)

    # Get Solver Function and Input Manager Function
    input_manager, solver = setupExperiment(args.input, args.solver)

    # Experiment Directory and create logger for store results
    path = str(args.input) + '/' + str(args.solver) + '/' + str(exp_config['experiment']['Seed'])
    file = str(exp_config['experiment']['Size']) + '-' + str(exp_config['experiment']['Initial_pos']) + '-' + str(exp_config['experiment'][
        'Env_Update'])

    logger = ExperimentLog(path, file)
    input = input_manager(exp_config,path,file)
    print(input)
    # Get Input Variables
    agent_pox_x = input[0][0]
    agent_pox_y = input[0][1]
    agent_pos = agent_pox_x, agent_pox_y
    nodes = input[1]
    Forest = input[2]
    time = input[3]
    max_budget = input[4]
    config = input[5]

    stats={}
    stats['env_type'] = exp_config['experiment']['env_type']
    stats['init_pos'] = int(agent_pox_x),int(agent_pox_y)
    stats['seed'] = exp_config['experiment']['Seed']
    stats['max_budget'] = max_budget


    # Call The solver
    tracing_start()
    start = tm.time()
    max_saved_trees, Hash_Calls, Hash = solver([agent_pox_x, agent_pox_y], nodes, Forest, time, max_budget, 0, config,0)
    end = tm.time()
    print("time elapsed {} milli seconds".format((end - start) * 1000))
    peak = tracing_mem()

    stats['per_time'] = (end - start) * 1000
    stats['per_mem'] = peak

    print("\nForest with:{n} nodes".format(n=len(nodes)))
    print("\nMax saved Trees:{t}".format(t=max_saved_trees))
    print("\nHash Repeated Calls:{h}".format(h=Hash_Calls))
    print("Hash Table has {l} different Forest Conditions".format(l=len(Hash)))

    # Construct Solution from Hash Table
    Solution = utils.Find_Solution(Forest, time, [agent_pox_x, agent_pox_y], Hash, config)
    print("\nSolution: {s}".format(s=Solution))

    sol_stats = [Solution, max_saved_trees, Hash, Hash_Calls]

    stats['sol'] = Solution
    stats['max_sav_trees'] = max_saved_trees
    stats['hash_calls'] = Hash_Calls

    logger.log_save(stats)

    # Build Path of actions that agent must follow considering DP Solution
    # Sol_Path = SolutionPath(Solution, input[0])
    # print("\nPath Solution of actions:\n{p}".format(p=Sol_Path))
