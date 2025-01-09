"""
computational_analysis.py

Provides scripts to analyze and compare different optimization methods, geometries, and/or lithography implementation
"""
import time

def comp_analysis(optimization_configs):
    """
    Run computational analyses for a list of optimizations, each with its own configuration.

    Parameters:
        optimization_configs (list of tuples): Each tuple contains an Optimization object and a dict of specific parameters.

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

    