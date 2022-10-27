import os

from dynaff.inputs import rndtree_metric
import numpy as np
import argparse
import sys
from pathlib import Path


class CreateGenerators:
    def __init__(self, grid, N, Load, defpath=None):
        self.N = N # Instances per Node Size
        self.Load = Load # Load seed of Experiment or Create a new one
        self.grid = grid # Tree Node size
        self.n_seeds = 0 # counter for total experiment seeds already created
        self.exp_selected = 0

        # If not Load, then generate a Seed_Sequence with  default entropy
        if self.Load in ('no', 'false', 'f', 'n', '0'):
            self.rnd_sq = np.random.SeedSequence()
            with open("Experiments/Seeds", 'r') as fp:
                self.n_seeds = len(fp.readlines())
            print("Generating New Seed Sequence: {s}".format(s=self.rnd_sq.entropy))
        # If we want to load a seed, then select from current seed directory to reproduce experiment
        else:
            print("Loading a Seed Sequence")
            if defpath is None:
                path = Path.cwd() / 'Experiments'
            else:
                path = Path(defpath)
            pathFile = path / "Seeds"
            if not pathFile.exists() or not pathFile.is_file():
                raise ValueError('Experiment Path either does not exists or is not a File')

            file = open(pathFile, 'rt')
            f = file.readlines()

            for line in f:
                print(line)
                self.n_seeds += 1
            file.close()

            while True:
                select = input("Insert index of desired seed to reproduce experiment: ")
                select = int(select)
                self.exp_selected = select
                if select >= 0 and select < self.n_seeds:
                    break
                print('Error: {} is not a valid option, please try again'.format(select))
            print("\nYou have selected experiment: {}".format(f[select]))

            self.grid = f[select].split()[2]
            self.N = int(f[select].split()[3])
            self.rnd_sq = np.random.SeedSequence(int(f[select].split()[1]))

        # Convert grid string (for load or not)
        grid_ = [int(element) for element in self.grid.split(",")]
        self.experiments = len(grid_)
        # Calculate number of instances to generate Generators
        self.total_instances = self.experiments * self.N

    def Create(self):
        # Spawn children for every instance
        children = self.rnd_sq.spawn(self.total_instances)
        # Create default generators for each instance
        generators = [np.random.default_rng(s) for s in children]
        return generators, self.rnd_sq, self.grid, self.N, self.n_seeds, self.exp_selected


def argParser(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''\
            Generate Experiment Instances or Load a seed to reproduce 
            previously created instances.
            ''',
        epilog='''python InstanceGenerator.py -l False -g 10,20,30,40 -s 5'''
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
    # Get input arguments
    args = argParser(sys.argv[:])
    # Instance Random Seed Generators
    Generator = CreateGenerators(args.grid, args.size, args.load)
    rnd_generators, sq, grid, N, n_seeds, exp_selected = Generator.Create()

    # Si se crea un nuevo experimento guardar semilla y parámetros
    if args.load in ('no', 'false', 'f', 'n', '0','False'):
        exp_string = str(n_seeds) + " " + str(sq.entropy) + " " +  str(grid) + " " + str(N) + "\n"
        fle = Path('Experiments/Seeds')
        fle.touch(exist_ok=True)
        seeds_file = open(fle,'a')
        seeds_file.write(exp_string)
        seeds_file.close()
        # Create Master Experiment Path if is new
        n_experiments = len(next(os.walk('Experiments'))[1])
        master_path = "Experiments/Experiment_" + str(n_experiments)
        if os.path.exists(master_path) == False:
            os.mkdir(master_path)
    else:
        master_path = "Experiments/Experiment_" + str(exp_selected)
        if os.path.exists(master_path) == False:
            os.mkdir(master_path)


    # Convert grid str to an array
    grid = [int(element) for element in grid.split(",")]

    # Save experiment Configuration parameters
    exp_config = {}
    exp_config['experiment'] = {}
    exp_config['experiment']['Env_Update'] = 1
    exp_config['experiment']['env_type'] = 'rnd_tree'
    exp_config['experiment']['env_metric'] = 'metric'
    exp_config['experiment']['instances'] = N

    # Configurable parameters for experiments
    exp_config['experiment']['scale'] = 10
    exp_config['experiment']['root_degree'] = 1
    # [A,B] We want agent at a distance between A% - B% of total scale
    exp_config['experiment']['delta'] = [.50,.75]
    ############################################

    c = 0
    for n_nodes in grid:
        # Create node_size folder
        node_path = master_path + "/" + 'Size_' + str(n_nodes) + '/'
        if not os.path.exists(node_path):
            os.mkdir(node_path)

        # Image name file
        file = 'img_' + str(n_nodes)

        # Partition Generators total/N
        batch_generators = rnd_generators[c:c + N]

        # Create instances with configuration and specific rnd_generators
        input = rndtree_metric(exp_config, node_path, file, n_nodes, batch_generators)

        c += 1