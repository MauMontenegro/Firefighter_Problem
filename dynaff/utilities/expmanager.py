from pathlib import Path
import yaml


def setupExperiment(input_type, solver_type):
    input_func, solver_func = startExperiment(input_type, solver_type)
    return input_func, solver_func


def startExperiment(input_type, solver_type):
    input = createInput(input_type)
    solver = createSolver(solver_type)
    return input, solver


def createSolver(solver):
    import dynaff.solvers as solvers
    target_class = solver
    print(solver)
    if hasattr(solvers, target_class):
        solverClass = getattr(solvers, target_class)
    else:
        raise AssertionError('There is no Solver called {}'.format(target_class))
    return solverClass


def createInput(input):
    import dynaff.inputs as inputs
    target_class = input
    if hasattr(inputs, target_class):
        inputClass = getattr(inputs, target_class)
    else:
        raise AssertionError('There is no Input Manager called {}'.format(target_class))
    return inputClass


def getExpConfig(name, defpath=None):
    if defpath is None:
        path = Path.cwd() / 'config'
    else:
        path = Path(defpath)
    pathFile = path / (name.strip() + '.yaml')

    if not pathFile.exists() or not pathFile.is_file():
        raise ValueError('Path either does not exists or is not a File')

    config = yaml.safe_load(pathFile.open('r'))

    return config
