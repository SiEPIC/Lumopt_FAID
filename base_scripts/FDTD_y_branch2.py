#############################################################################
# Python module: FDTD_y_branch.py 
#
# Description:
# This module defines the y_brach_init_() function used in the 
# optimization for inverse design of the SOI Y-branch in 3D
#
# Steps include:
# 1. Define the base simulation parameters 
# 2. Define the geometry of input and output waveguides
# 3.Set up source and monitors and simulation region
# 
# Copyright 2019, Lumerical Solutions, Inc.
##############################################################################

######## IMPORTS ########
# General purpose imports
import lumapi
import numpy as np
from scipy.constants import c

settings = {
    'size_x': 3e-6,
    'size_y': 3e-6,
    'mesh_x': 20e-9,
    'mesh_y': 20e-9,
    'finer_mesh_size': 2.5e-6,
    'mesh_accuracy': 2,
}

def update_settings(key, value):
    """Update configuration settings."""
    global settings
    if key in settings:
        settings[key] = value
    else:
        raise KeyError(f"Key {key} not found in settings.")
    
def y_branch_init_(fdtd): 
    fdtd.switchtolayout()
    fdtd.selectall()
    fdtd.delete()
    
    # Settings Params
    size_x = settings['size_x']
    size_y = settings['size_y']
    mesh_x = settings['mesh_x']
    mesh_y = settings['mesh_y']
    finer_mesh_size = settings['finer_mesh_size']
    mesh_accuracy = settings['mesh_accuracy']
    
    ## GEOMETRY
    
    # INPUT WAVEGUIDE
    fdtd.addrect()
    fdtd.set('name', 'input wg')
    fdtd.set('x span', 3e-6)
    fdtd.set('y span', 0.5e-6)
    fdtd.set('z span', 220e-9)
    fdtd.set('y', 0)
    fdtd.set('x', -2.5e-6)
    fdtd.set('index', 2.8)
    
    # OUTPUT WAVEGUIDES
    fdtd.addrect()
    fdtd.set('name', 'output wg top')
    fdtd.set('x span', 3e-6)
    fdtd.set('y span', 0.5e-6)
    fdtd.set('z span', 220e-9)
    fdtd.set('y', 0.35e-6)
    fdtd.set('x', 2.5e-6)
    fdtd.set('index', 2.8)
    
    fdtd.addrect()
    fdtd.set('name', 'output wg bottom')
    fdtd.set('x span', 3e-6)
    fdtd.set('y span', 0.5e-6)
    fdtd.set('z span', 220e-9)
    fdtd.set('y', -0.35e-6)
    fdtd.set('x', 2.5e-6)
    fdtd.set('index', 2.8)
    
    ## SOURCE
    fdtd.addmode()
    fdtd.set('direction', 'Forward')
    fdtd.set('injection axis', 'x-axis')
    #fdtd.set('polarization angle',0)
    fdtd.set('y', 0.0)
    fdtd.set('y span', size_y)
    fdtd.set('x', -1.25e-6)
    fdtd.set('override global source settings', False)
    fdtd.set('mode selection', 'fundamental TE mode')
    
    ## FDTD
    fdtd.addfdtd()
    fdtd.set('dimension', '2D')
    fdtd.set('background index', 1.44)
    fdtd.set('mesh accuracy', mesh_accuracy)
    fdtd.set('x', 0.0)
    fdtd.set('x span', size_x)
    fdtd.set('y', 0.0)
    fdtd.set('y span', size_y)
    fdtd.set('force symmetric y mesh', True)
    fdtd.set('y min bc', 'Anti-Symmetric')
    fdtd.set('pml layers', 12)
    
    ## MESH IN OPTIMIZABLE REGION
    fdtd.addmesh()
    fdtd.set('x', 0)
    fdtd.set('x span', finer_mesh_size + 2.0 * mesh_x)
    fdtd.set('y', 0)
    fdtd.set('y span', finer_mesh_size)
    fdtd.set('dx', mesh_x)
    fdtd.set('dy', mesh_y)
    
    ## OPTIMIZATION FIELDS MONITOR IN OPTIMIZABLE REGION
    fdtd.addpower()
    fdtd.set('name', 'opt_fields')
    fdtd.set('monitor type', '2D Z-normal')
    fdtd.set('x', 0)
    fdtd.set('x span', finer_mesh_size)
    fdtd.set('y', 0)
    fdtd.set('y span', finer_mesh_size)
    
    ## FOM FIELDS
    fdtd.addpower()
    fdtd.set('name', 'fom')
    fdtd.set('monitor type', '2D X-normal')
    fdtd.set('x', finer_mesh_size / 2.0)
    fdtd.set('y', 0.0)
    fdtd.set('y span', size_y)