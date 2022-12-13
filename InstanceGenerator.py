import os

from dynaff.inputs import rndtree_metric
import numpy as np
import argparse
import sys
from pathlib import Path


class CreateGenerators:
    """
    Create Random Number Generators

    :param N: Number of Instances generated
    :param Load: Selection fLag for load on experiment or create one
    :param grid: Array of Total Tree Nodes [10 20 30]
    :param n_seeds: Number of existing experiment seeds
    :param exp_selected: Selected Experiment
    """
    def __init__(self, grid, N, Load, defpath=None):
        self.N = N
        self.Load = Load
        self.grid = grid
        self.n_seeds = 0
        self.exp_selected = 0

        if defpath is None:
            path = Path.cwd() / 'Experiments' / "Seeds"
        else:
            path=Path(defpath)

        # Generate a Seed_Sequence with  default entropy
        if self.Load in ('no', 'false', 'f', 'n', '0'):
            self.rnd_sq = np.random.SeedSequence()
            with open(path, 'r') as fp:
                self.n_seeds = len(fp.readlines())
            print("Generating New Seed Sequence: {s}".format(s=self.rnd_sq.entropy))
        # Select from current seed directory to reproduce experiment
        else:
            print("Loading a Seed Sequence")
            if not path.exists() or not path.is_file():
                raise ValueError('Experiment Path either does not exists or is not a File')
            file = open(path, 'rt')
            # Display Seeds with their selection index
            f = file.readlines()
            for line in f:
                print(line)
                self.n_seeds += 1
            file.close()

            while True:
                select = int(input("Insert index for desired seed to reproduce experiment: "))
                self.exp_selected = select
                if select >= 0 and select < self.n_seeds:
                    break
                print('Error: {} is not a valid option, please try again'.format(select))
            print("\nYou have selected experiment: {}".format(f[select]))

            self.grid = f[select].split()[2]
            self.N = int(f[select].split()[3])
            self.rnd_sq = np.random.SeedSequence(int(f[select].split()[1]))

        grid_ = [int(element) for element in self.grid.split(",")]
        self.experiments = len(grid_)
        self.total_instances = self.experiments * self.N

    def Create(self):
        children = self.rnd_sq.spawn(self.total_instances)          # Spawn children for every instance
        generators = [np.random.default_rng(s) for s in children]   # Create default generators for each instance
        return generators, self.rnd_sq, self.grid, self.N, self.n_seeds, self.exp_selected


def argParser(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''\
            Generate Experiment Instances or Load a seed to reproduce 
            previously created instances.
            ''',
        epilog='''python InstanceGenerator.py -l f -g 10,20,30,40 -s 5'''
    )

    parser.add_argument(
        '--load', '-l', type=str,
        help="Load Experiments or Create New One")
    parser.add_argument(
        '--grid', '-g', type=str,
        help="Grid of experiment node sizes for Tree")
    parser.add_argument(
        '--size', '-s', type=int,
        help="Number o instances for each node size")

    return parser.parse_known_args(args)[0]


if __name__ == '__main__':
    args = argParser(sys.argv[:])
    Generator = CreateGenerators(args.grid, args.size, args.load)
    rnd_generators, sq, grid, N, n_seeds, exp_selected = Generator.Create()

    # Create New Experiment and save seed for reproducibility
    if args.load in ('no', 'false', 'f', 'n', '0','False'):
        # Experiment save example: [5   3435464535  10,20,30    10]
        exp_string = str(n_seeds) + " " + str(sq.entropy) + " " +  str(grid) + " " + str(N) + "\n"
        fle = Path('Experiments/Seeds')
        # Create seed file if not exist
        fle.touch(exist_ok=True)
        # Write Experiment string
        seeds_file = open(fle,'a')
        seeds_file.write(exp_string)
        seeds_file.close()
        # Create Master Experiment Path if is new
        n_experiments = len(next(os.walk('Experiments'))[1]) # Get number of next number of experiment
        master_path = "Experiments/Experiment_" + str(n_experiments)
        if os.path.exists(master_path) == False:
            os.mkdir(master_path)
    else:
        master_path = "Experiments/Experiment_" + str(exp_selected)
        if os.path.exists(master_path) == False:
            os.mkdir(master_path)


    # Convert grid string to array
    grid = [int(element) for element in grid.split(",")]

    # Experiment Environment Parameters
    exp_config = {}
    exp_config['experiment'] = {}
    exp_config['experiment']['env_type'] = 'rnd_tree'
    exp_config['experiment']['env_metric'] = 'metric'
    exp_config['experiment']['instances'] = N
    # Configurable Experiment Parameters
    exp_config['experiment']['scale'] = 1
    exp_config['experiment']['root_degree'] = 1
    exp_config['experiment']['Env_Update'] = 1
    exp_config['experiment']['delta'] = [.50,.75]  # [A,B] We want agent at a distance between A% - B% of total scale
    ############################################
    c = 0
    # Create N instances for each Tree Size Experiment in Grid
    for n_nodes in grid:
        node_path = master_path + "/" + 'Size_' + str(n_nodes) + '/'
        if not os.path.exists(node_path):
            os.mkdir(node_path)
        file = 'img_' + str(n_nodes)
        batch_generators = rnd_generators[c:c + N] # Partition Generators total/N
        input = rndtree_metric(exp_config, node_path, file, n_nodes, batch_generators)
        c += N