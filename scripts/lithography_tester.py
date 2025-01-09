"""
lithography_tester.py

Script to analyze lithography-optimized inverse design
"""
import time
from config import Config
from lumopt_gl.utilities.lithography import LithographyOptimizer
import copy

def lithography_tester(opt, lithography_model, **kwargs):
    """
    Compares the lithography output of a litho optimized and unoptimized geometry. 

    Parameters:
        opt (object): an optimization object that has NOT been initialized.
        lithography_model (str): lithography model to test. 
    
    **kwargs:
            - stop_when_success (bool): a 'success' is defined as when the litho-optimized geometry
                                        performs better than the litho output of the unoptimized geometry.
            - result_name (str): folder within the results folder to save the results
            

    Returns:
        opts: A dict containing two keys: optimized (for lithography), unoptimized.
              Each value is a list containing 5 elements:
              Pre-litho FOM, Post-litho FOM, Parameters, Time
    """
    result_dict = {}
    # produce a hash I guess for saving.
    # extract the base parameters from opt. These will be used to reset the optimizer for lithography
    litho_opt = copy.deepcopy(opt)
    opt.geometry.lithography_model = None
    start_time = time.time()
    results = opt.run(working_dir=f'{Config.RESULTS_PATH}/lithography_test_unoptimized')
    unoptimized_time = time.time() - start_time
    opt.geometry.lithography_model = LithographyOptimizer(model=lithography_model)
    unoptimized_with_litho = opt.run_forward_simulation()
    result_dict['optimized'] = [results[0], unoptimized_with_litho, results[1], unoptimized_time]
    
    # need to create a helper function to reset params if we need to
    start_time = time.time()
    litho_opt.geometry.lithography_model = LithographyOptimizer(model=lithography_model)
    litho_results = litho_opt.run(working_dir=f'{Config.RESULTS_PATH}/lithography_test_unoptimized')
    litho_opt.geometry.lithography_model = None
    optimized_without_litho = litho_opt.run_forward_simulation()
    optimized_time = time.time() - start_time
    result_dict['unoptimized'] = [optimized_without_litho, litho_results[0], results[1], optimized_time]
    
    return result_dict

def lithography_visualizer(opts, **kwargs):
    """
    Takes a litho optimized object and an unoptimized object and produces the relevant graphs.

    Parameters:
        opt (object): an optimization object that has NOT been initialized.
        lithography_model (str): lithography model to test. 
    
    **kwargs:
            - stop_when_success (bool): a 'success' is defined as when the litho-optimized geometry
                                        performs better than the litho output of the unoptimized geometry.
            - time (bool): flag to return time required to optimize

    Returns:
        list: A list of results from each optimization run.
    """
    results = []
    durations = []
    for item in optimization_configs:
        # Check if the item is a tuple with configuration or just an optimization object
        if isinstance(item, tuple):
            opt, config = item
            if 'method' in config:
                opt.optimizer.method = config['method']
            if 'max_iter' in config:
                opt.optimizer.max_iter = config['max_iter']
            if 'lithography_model' in config:
                opt.geometry.lithography_model = config['lithography_model']
        else:
            opt = item

        # Run the optimization
        start_time = time.time()
        result = opt.run()
        end_time = time.time()
        
        results.append(result)
        durations.append(end_time - start_time)

    return [results, durations]
