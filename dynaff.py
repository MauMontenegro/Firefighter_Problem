import argparse
import sys
import numpy as np
from matplotlib import pyplot as plt
import json
from pathlib import Path
import tracemalloc
import time as tm
import os

from ete3 import Tree
import networkx as nx

from dynaff import setupExperiment, getExpConfig
from dynaff.utilities import utils
from dynaff.inputs.rndtree_metric import TreeConstruct

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
        epilog='''python dynaff.py -s dpsolver -i rndtree -e experiment_name'''
    )

    parser.add_argument(
        '--input', '-i', type=str,
        help="Type of Input for Solver.")
    parser.add_argument(
        '--solver', '-s', type=str,
        help="Type of Solver.")
    parser.add_argument(
        '--experiment', '-e', type=str,
        help="Config File.")

    return parser.parse_known_args(args)[0]

def saveSolution(instance_path,instance,solution,saved,time,hash_calls,hash_size,solver_name):
    summary_name=solver_name + "_summary"
    output_path = instance_path / instance / summary_name
    with open(output_path, "w") as writer:
        writer.write("Solution: {}\n".format(solution))
        writer.write("Saved: {}\n".format(saved))
        writer.write("RunTime: {}\n".format(time))
        writer.write("Hash_Calls: {}\n".format(hash_calls))
        writer.write("Hash Size: {}\n".format(hash_size))

def Statistics(path,total_saved,total_times):
    time_mean = []
    time_std_dv=[]
    saved_mean = []
    saved_std_dv = []

    # Statistics for run time
    for node_size in total_times:
        m = np.mean(node_size)
        std = np.std(node_size)
        time_mean.append(m)
        time_std_dv.append(std)
    time_std_dv = np.asarray(time_std_dv)
    time_mean = np.asarray(time_mean)

    # Statistics for saved vertices
    for node_size in total_saved:
        m = np.mean(node_size)
        std = np.std(node_size)
        saved_mean.append(m)
        saved_std_dv.append(std)
    saved_std_dv=np.asarray(saved_std_dv)
    saved_mean=np.asarray(saved_mean)

    print(saved_mean)
    print(saved_std_dv)


    np.save(path / "Statistics_DP", np.array([saved_mean, saved_std_dv, time_mean, time_std_dv]))
    y= np.arange(0, len(time_mean), 1, dtype=int)
    fig, ax = plt.subplots(1)
    ax.plot(y,saved_mean, label="Mean saved Vertices",color="blue")
    ax.fill_between(y, saved_mean+saved_std_dv/2,saved_mean-saved_std_dv/2,facecolor="blue",alpha=0.5)
    plt.savefig(path / 'DP_Saved.png')

    fig, ax = plt.subplots(1)
    ax.plot(y, time_mean, label="Mean Time Vertices", color="red")
    ax.fill_between(y, time_mean + time_std_dv/2, time_mean - time_std_dv/2, facecolor="red", alpha=0.5)
    plt.savefig(path / 'DP_Time.png')

if __name__ == '__main__':
    #Get input arguments and handle missing values
    args = argParser(sys.argv[:])
    if args.solver == None:
        raise argparse.ArgumentTypeError('No solver selected')
    if args.experiment == None:
        raise argparse.ArgumentTypeError('No Experiment folder selected')
    if args.input == None:
        raise argparse.ArgumentTypeError('No input selected')

    #Experiment path
    path = Path.cwd() / "Experiments" / str(args.experiment)
    # Get Solver Function and Input Manager Function
    input_manager, solver = setupExperiment(args.input, args.solver)

    total_times=[]
    total_saved=[]
    size_dirs = []
    for d in next(os.walk(path)):
        size_dirs.append(d)
    size_dirs = sorted(size_dirs[1])

    # Traverse for each Tree Size experiments
    for dir in size_dirs:
        instance_path = path / str(dir)
        inst_dirs = []
        for i in next(os.walk(instance_path)):
            inst_dirs.append(i)
        inst_dirs = sorted(inst_dirs[1])
        saved_p_nodes = []
        t_p_nodes = []
        # Traverse for each Instance
        for inst in inst_dirs:
            print("\n\n>>>>>>Compute solution for {n}, {i} <<<<<<<<".format(n=dir, i=inst))
            # Load Instance
            T, N, starting_fire, T_Ad_Sym, seed, scale, agent_pos, max_degree, root_degree, time = \
                utils.generateInstance(True, instance_path, str(inst))
            # Generate Tree
            F = Tree()                                  # Tree in ete3 form
            all_nodes = {}                              # Node Dictionary
            F, all_nodes, max_level, levels, _, _ = TreeConstruct(F, all_nodes, T, starting_fire)
            # Get node Levels
            levels_ = nx.single_source_shortest_path_length(T, starting_fire)
            nx.set_node_attributes(T, levels_, "levels")
            marked_list = [0] * T.number_of_nodes()
            nx.set_node_attributes(T, marked_list, "marked")
            # CALL THE SOLVER
            # ----------------------------------------------------------------------------------------------------------
            tracing_start()
            start = tm.time()
            max_saved_trees, Sol = solver(agent_pos, all_nodes, time, time, 0, T_Ad_Sym, 0, T)
            end = tm.time()
            t = (end - start)
            print("time elapsed {} seconds".format(t))
            peak = tracing_mem()
            # -----------------------------------------------------------------------------------------------------------
            # Just Printing Results
            print("\nForest with:{n} nodes".format(n=len(all_nodes)))
            print("Max saved Trees:{t}".format(t=max_saved_trees))
            #print("Hash Repeated Calls:{h}".format(h=Hash_Calls))
            print("Hash Table has {l} different Forest Conditions".format(l=len(Sol)))
            # Retrieve Solution Strategy
            if args.solver == "dpsolver_mau": # Dynamic Programming
                Sol = utils.Find_Solution(F, time, agent_pos, Sol, T_Ad_Sym)
                Sol.insert(0, str(N))
                print("\nSolution Sequence: {s}".format(s=Sol))
            if args.solver == "hd_heuristic" or args.solver == "ms_heuristic": # Heuristics
                print("\n Solution: {s}".format(s=Sol[0]))
                print("\n Time elapsed by step: {s}".format(s=Sol[1]))
                print("\n Fireline Level: {s}".format(s=Sol[2]))
                # utils.graphSolution(Sol, plotting)
            Hash_Calls = 0
            # Saving stats for general parameters
            saveSolution(instance_path, inst, Sol, max_saved_trees, t, Hash_Calls, len(Sol),args.solver)
            # Saved nodes per Graph
            saved_p_nodes.append(max_saved_trees)
            t_p_nodes.append(t)
        total_times.append(t_p_nodes)
        total_saved.append(saved_p_nodes)
    Statistics(path, total_saved, total_times)
