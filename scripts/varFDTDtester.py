"""
varFDTD_tester.py

Tests the integrity of varFDTD for a certain geometry by comparing results with FDTD. 
"""
import random
import numpy as np
import matplotlib.pyplot as plt

def varFDTD_tester(opt2D, opt3D, params_list = None):
    """
    Takes a 2D and a 3D optimization object, and an optional list of parameters. Returns the difference in FOM for each set of params

    Parameters:
        opt2D (object): an optimization object using varFDTD with a 2D basescript
        opt3D (object): an optimization object using FDTD with a 3D basescript
        params_list (list): a list of parameters to test. If none are provided the script will use the base parameters
                     and then adjust them randomly 10 times
        

    Returns:
        result_dict (dict): a dict containing:
                            difference: the percentage difference between varFDTD and FDTD results
                            2D_results: the varFDTD results for each set of parameters
                            3D_results: the FDTD results for each set of parameters
                            parameters: list of parameters tested.
    """
    
    opt2D.initialize()
    opt3D.initialize()
    
    if params_list is None:
        # If no parameters are provided, use a base set and create random variations
        try:
            base_params = opt2D.geometry.params  # Replace with your actual base parameters
        except:
            base_params = opt2D.geometry.current_params  # Replace with your actual base parameters
        params_list = [base_params]
        
        for _ in range(10):
            random_params = [param * random.uniform(0.8, 1.25) for param in base_params]
            params_list.append(random_params)
        
    else:
        # If parameters are provided, use them directly
        params_list = params_list
        
    varFDTD_results = []
    FDTD_results = []
    
    for params in params_list:
        # Run simulations for both 2D and 3D with the given parameters
        varFDTD_result = opt2D.run_forward_simulation(params)
        FDTD_result = opt3D.run_forward_simulation(params)
        
        varFDTD_results.append(varFDTD_result)
        FDTD_results.append(FDTD_result)
    
    # Calculate the percentage difference between varFDTD and FDTD results
    differences = []
    for var_res, fdtd_res in zip(varFDTD_results, FDTD_results):
        if fdtd_res != 0:  # Avoid division by zero
            difference = abs(var_res - fdtd_res) / abs(fdtd_res) * 100
        else:
            difference = float('inf')  # Indicate infinite difference if FDTD result is zero
        differences.append(difference)
    
    # Prepare the result dictionary
    result_dict = {
        'difference': differences,
        '2D_results': varFDTD_results,
        '3D_results': FDTD_results,
        'parameters': params_list
    }

    return result_dict
    
def varFDTD_convergence_tester(opt2D, opt3D):
    """
    Takes a 2D and a 3D optimization object. Returns the params and FOM the solutions converged to.

    Parameters:
        opt2D: an optimization object using varFDTD with a 2D basescript
        opt3D: an optimization object using FDTD with a 3D basescript

    Returns:
        result_dict: a dict containing:
                    difference: the percentage difference between varFDTD and FDTD results
                    2D_results: the varFDTD results for each set of parameters
                    3D_results: the FDTD results for each set of parameters
                    parameters: list of parameters tested.
    """
    
    opt2D.initialize()
    opt3D.initialize()
    
    varFDTD_result = opt2D.run()
    FDTD_result = opt3D.run()
    
    FDTD_result_on_varFDTD = opt2D.run_forward_solves(np.array(FDTD_result[1]))
    varFDTD_result_on_FDTD = opt3D.run_forward_solves(np.array(varFDTD_result[1]))
    # Calculate the percentage difference between varFDTD and FDTD results
    
    results = [varFDTD_result, FDTD_result, FDTD_result_on_varFDTD, varFDTD_result_on_FDTD]

    return results
    
def varFDTD_visualizer(opt2D, opt3D):
    """
    Runs the 3D simulations based on the 2D optimization parameter history, and plots the Figure of Merit (FOM) history for both 2D and 3D optimizations.

    Parameters:
        opt2D: An optimization object using varFDTD with a 2D basescript.
        opt3D: An optimization object using FDTD with a 3D basescript.
    """
    opt2D.run()
    opt3D.initialize()
    
    param_hist = opt2D.params_hist
    fom_2D_hist = opt2D.fom_hist
    
    fom_3D_hist = []
    
    for param in param_hist:
        fom_3D_hist.append(opt3D.run_forward_simulation(np.array(param)))
    
    fom_2D_hist = np.array(fom_2D_hist) * -1
    
    plt.plot(fom_2D_hist, label="2D FOM")
    plt.plot(fom_3D_hist, label="3D FOM", linestyle="--")
    plt.xlabel("Iteration")
    plt.ylabel("Figure of Merit (FOM)")
    plt.title("2D vs 3D FOM History")
    plt.legend()
    plt.show()
    
    return [fom_2D_hist, fom_3D_hist]