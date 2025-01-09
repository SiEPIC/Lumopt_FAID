"""
topology_geos.py

Topology optimization doesn't use polygons and geometries but rather a design region. 

As a result, this file mostly provides scripts to initialize basescripts for several topology optimization cases and enables
automatic adjustment of the basescripts and initial guesses.

"""

def y_branch_topology_basescript(dimension = 2, **kwargs):
    """
    Returns the ybranch basescript for topology optimization.

    Parameters:
        dimension (int): 2 sets up a varFDTD simulation, 3 for FDTD (default)
        
        **kwargs:
            - 'design_region_x/design_region_y' (float): x and y spans of the opt region
            - 'initial_guess' (bool): Flag to use initial guess connecting the input and output waveguide.
            - 'mesh' (int): mesh size for simulation. 
    Returns:
        base_script: modified base_script based on the kwargs provided
    """
    # else, create a dynamic base script based on kwargs
    if dimension  == 2:
        from base_scripts.FDTD_y_branch_topology import y_branch_2D_topo_init, update_settings
        for key, value in kwargs.items():
            update_settings(key, value)
        base_script = y_branch_2D_topo_init
    if dimension  == 2:
        from base_scripts.FDTD_y_branch_topology import y_branch_3D_topo_init, update_settings
        for key, value in kwargs.items():
            update_settings(key, value)
        base_script = y_branch_3D_topo_init
        
    return base_script

