#############################################################################
# Python Module: varFDTD_y_branch.py  
#
# Description:
# This module defines the y_brach_init_() function used in the 
# optimization for inverse design of the SOI Y-branch in 2D
#
# Steps include:
# 1. Define the base simulation parameters 
# 2. Add non-dispersive materials 
# 2. Define the geometry of input and output waveguides
# 3. Set up source and monitors and simulation region
# 
# Copyright 2019, Lumerical Solutions, Inc.
##############################################################################

######## IMPORTS ########
# General purpose imports
import lumapi
import numpy as np
from scipy.constants import c

# settings dict allows for convenient adjustment of key parameters at base script initialization
settings = {
    'start_x': -1e-6,
    'stop_x': 1e-6,
    'start_y': 0.25e-6,
    'stop_y': 0.6e-6,
    'min_bound': 0.2e-6,
    'max_bound': 0.8e-6,
    'poly_wg': False,
    'num_params': 10,
    'mesh': 4,
}

def update_settings(key, value):
    """Update configuration settings for the Y-branch simulation."""
    global settings
    if key in settings:
        settings[key] = value
    else:
        raise KeyError(f"Key {key} not found in settings.")
    
    
def taper_init_(mode): 
           
    ## CLEAR SESSION
    mode.switchtolayout()
    mode.selectall()
    mode.delete()
    
    x_start = settings['start_x']
    x_end = settings['stop_x']
    y_start = settings['start_y']
    y_end = settings['stop_y']
    poly_wg = settings['poly_wg']
    wg_width_in = 2 * y_start
    wg_width_out = 2 * y_end
    mesh_accuracy = settings['mesh']
    
    ## SIM PARAMS
    size_x = (x_end - x_start) * 1.5  # slightly larger than the design region
    size_y = size_x
    mesh_x = 20e-9
    mesh_y = 20e-9
    finer_mesh_size= (x_end - x_start) + 0.4e-6
    lam_c = 1.550e-6  # will be overridden from opt if necessary
    
    ## MATERIAL
    opt_material = mode.addmaterial('Dielectric')
    mode.setmaterial(opt_material, 'name', 'Si: non-dispersive')
    n_opt = mode.getindex('Si (Silicon) - Palik', c / lam_c)
    mode.setmaterial('Si: non-dispersive', 'Refractive Index', n_opt)
    
    sub_material = mode.addmaterial('Dielectric')
    mode.setmaterial(sub_material, 'name', 'SiO2: non-dispersive')
    n_sub = mode.getindex('SiO2 (Glass) - Palik', c / lam_c)
    mode.setmaterial('SiO2: non-dispersive', 'Refractive Index', n_sub)
    mode.setmaterial('SiO2: non-dispersive', "color", np.array([0, 0, 0, 0]))
    
    ## GEOMETRY
    
    if not poly_wg:  # if we aren't passing the waveguides as polygons, then they are static.
        
        mode.addrect()
        mode.set('name', 'input wg')
        mode.set('x span', 3e-6)
        mode.set('y span', wg_width_in)
        mode.set('z span', 220e-9)
        mode.set('y', 0)
        mode.set('x', x_start - 1.5e-6)
        mode.set('z', 0)
        mode.set('material', 'Si: non-dispersive')
        
        mode.addrect()
        mode.set('name', 'output wg')
        mode.set('x span', 3e-6)
        mode.set('y span', wg_width_out)
        mode.set('z span', 220e-9)
        mode.set('y', 0)
        mode.set('x', x_end + 1.5e-6)
        mode.set('z', 0)
        mode.set('material', 'Si: non-dispersive')
    
    mode.addrect()
    mode.set('name', 'sub')
    mode.set('x span', 2 * ((x_end - x_start) + 2e-6))  # end of design region + 3e-6 for the output waveguide
    mode.set('y span', 2 * ((x_end - x_start) + 2e-6))
    mode.set('z span', 10e-6)
    mode.set('y', 0)
    mode.set('x', 0)
    mode.set('z', 0)
    mode.set('material', 'SiO2: non-dispersive')
    mode.set('override mesh order from material database', 1)
    mode.set('mesh order', 3)
    mode.set('alpha', 0.8)
    
    ## varFDTD
    mode.addvarfdtd()
    mode.set('mesh accuracy', mesh_accuracy)
    mode.set('x min', -size_x / 2)
    mode.set('x max', size_x / 2)
    mode.set('y min', -size_y / 2)
    mode.set('y max', size_y / 2)
    mode.set('force symmetric y mesh', 1)
    mode.set('y min bc', 'Anti-Symmetric')
    mode.set('z', 0)
    
    mode.set('effective index method', 'variational')
    mode.set('can optimize mesh algorithm for extruded structures', 1)
    mode.set('clamp values to physical material properties', 1)
    
    mode.set('x0', -1.3e-6)
    mode.set('number of test points', 4)
    mode.set('test points', np.array([[0, 0], [1.3e-6, 0.4e-6], [1.3e-6, -0.4e-6], [1.3e-6, 0]]))
    
    ## SOURCE
    mode.addmodesource()
    mode.set('direction', 'Forward')
    mode.set('injection axis', 'x-axis')
    # mode.set('polarization angle',0)
    mode.set('y', 0)
    mode.set('y span', size_y)
    mode.set('x', x_start - 0.2e-6)
    mode.set('center wavelength', 1550e-9)
    mode.set('wavelength span', 0)
    mode.set('mode selection', 'fundamental mode')
    
    
    ## MESH IN OPTIMIZABLE REGION
    mode.addmesh()
    mode.set('x', 0)
    mode.set('x span', finer_mesh_size)
    mode.set('y', 0)
    mode.set('y span', finer_mesh_size)
    mode.set('dx', mesh_x)
    mode.set('dy', mesh_y)
    
    ## OPTIMIZATION FIELDS MONITOR IN OPTIMIZABLE REGION
    
    mode.addpower()
    mode.set('name', 'opt_fields')
    mode.set('monitor type', '2D Z-normal')
    mode.set('x', 0)
    mode.set('x span', finer_mesh_size)
    mode.set('y', 0)
    mode.set('y span', finer_mesh_size)
    mode.set('z', 0)
    
    ## FOM FIELDS
    
    mode.addpower()
    mode.set('name', 'fom')
    mode.set('monitor type', 'Linear Y')
    mode.set('x', x_end + 0.20e-6)
    mode.set('y', 0)
    mode.set('y span', size_y)
    mode.set('z', 0)

