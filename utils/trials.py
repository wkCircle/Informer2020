from typing import List
from pathlib import Path 
import pickle 
import hyperopt
from hyperopt.fmin import generate_trials_to_calculate

def load_trials(
    path2file: str, 
    force_new=False, 
    points: List[dict]=[], 
    verbose=True
): 
    """
    load trials from a ``path2file`` str path if exists. Otherwise, always return a newly created Trials object of Hyperopt with initial ``points`` assigned.

    Args:
        path2file (str): The path to the file nmae, should be of pickle type (.pkl). 
        force_new (bool): If true, then always return new Trials with ``points``
            initialized when ``points`` is not empty. 
        
        points (List[dict]): A list of dict where given points are evaluated 
            before optimisation starts, so the overall number of optimisation 
            steps is len(points_to_evaluate) + max_evals. Elements of this list 
            must be in a form of a dictionary with variable names as keys and 
            variable values as dict values. Example points value is 
            [{'x': 0.0, 'y': 0.0}, {'x': 1.0, 'y': 2.0}]. Defaults to []. 
        
        verbose (bool): print message to concole or not. 

    Returns:
        hyperopt.Trials: a loaded Trials object if path2file exists, otherwise a new Trials object with ``points`` initialized is returned. 
    """
    if force_new: 
        return generate_trials_to_calculate(points=points)
    
    fp = Path(path2file)
    if fp.is_file(): 
        if verbose: 
            print(f"Continue hyperopt training since last time from {fp}")
        with fp.open('rb') as fr: 
            trials = pickle.load(fr)
    else: 
        trials = generate_trials_to_calculate(points=points)
    return trials 