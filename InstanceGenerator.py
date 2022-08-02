from dynaff.inputs import rndtree_metric
import os

if __name__ == '__main__':
    grid = [10, 20, 30, 40, 50, 60]
    exp_config={}
    exp_config['experiment'] = {}
    exp_config['experiment']['Seed'] = 600
    exp_config['experiment']['Env_Update'] = 1
    exp_config['experiment']['env_type'] = 'rnd_tree'
    exp_config['experiment']['env_metric'] = 'metric'
    exp_config['experiment']['scale'] = 1


    for n_nodes in grid:
        # Create path and file name
        path = 'Instances/' + 'Instance_' + str(n_nodes) + '/'
        if os.path.exists('Instances/' + 'Instance_' + str(n_nodes)) == False:
            os.mkdir('Instances/' + 'Instance_' + str(n_nodes))
        file = 'img_' + str(n_nodes)

        # Get Instance for Selected Solver and config File.
        input = rndtree_metric(exp_config, path, file, n_nodes)
        agent_pos_x = input[0][0]
        agent_pos_y = input[0][1]
        agent_pos = agent_pos_x, agent_pos_y
        nodes = input[1]
        Forest = input[2]
        time = input[3]
        max_budget = input[4]
        config = input[5]
        plotting = input[6]
