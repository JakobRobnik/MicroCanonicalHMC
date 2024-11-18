

from itertools import product
import os, sys, inspect
import pandas as pd



def base_dir(param_grid):
    folder = 'ensemble/img/'
    for name in param_grid.keys():
        folder += name + '_'
    return folder[:-1] + '/'


def subdir(values):
    file = ''
    for val in values:
        file += str(val) + '_'
    return file[:-1] + '/'
    

def do_grid(func, param_grid, fixed_params=None, verbose= False):
    """
    Perform a grid search over specified parameters for a given function, 
    while keeping other parameters fixed to their default values, and allowing
    for parameters without default values to be specified separately.

    Parameters:
    - func: The function to evaluate. It must return a dictionary.
    - param_grid: A dictionary where keys are parameter names and values are lists of values to try.
    - fixed_params: A dictionary of parameters that don't have default values, to be fixed across all evaluations.

    Returns:
    - A pandas DataFrame where each row represents a parameter combination,
      excluding fixed parameters, with additional columns corresponding to the
      keys of the dictionary returned by the function.
    """
    
    base = base_dir(param_grid)

    if not os.path.isdir(base):
        os.mkdir(base)

    # Get the function's default parameter values
    sig = inspect.signature(func)
    default_params = {
        k: v.default for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty
    }

    # Combine default and fixed parameters
    fixed_params = fixed_params or {}
    all_fixed_params = {**default_params, **fixed_params}

    # Prepare the grid of parameter combinations
    grid_keys = list(param_grid.keys())
    grid_values = list(param_grid.values())
    combinations = list(product(*grid_values))
    
    results = []
    counter = 0
    for values in combinations:
        if verbose: print(f'{counter} / {len(combinations)}')
        counter += 1
        # Update parameters for this combination
        params = all_fixed_params.copy()
        params.update(dict(zip(grid_keys, values)))
        # Evaluate the function with the current parameter set
        dir = base + subdir(values)
        params['dir'] = dir
        
        if not os.path.isdir(dir):
            os.mkdir(dir)

        result_dict = func(**params)
        if not isinstance(result_dict, dict):
            raise ValueError("The function must return a dictionary.")
        
        # Prepare a row with the varying parameters and result dictionary values
        row = {
            k: params[k] for k in sig.parameters.keys()
        }
        row.update(result_dict)  # Add the function output dictionary
        results.append(row)
    
    # Convert results to a pandas DataFrame
    df = pd.DataFrame(results)
    df = df.drop(columns= ['dir', ])
    
    df.to_csv(base + 'data.csv', sep= '\t', index= False)