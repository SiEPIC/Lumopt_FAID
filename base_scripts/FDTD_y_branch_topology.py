######## IMPORTS ########
# General purpose imports
import lumapi
import numpy as np
from scipy.constants import c

settings = {
    'design_region_x': 3.5e-6,
    'design_region_y': 3.5e-6,
    'initial_guess': True,
    'mesh': 3,
    }

def update_settings(key, value):
    """Update configuration settings for the Y-branch simulation."""
    global settings
    if key in settings:
        settings[key] = value
    else:
        raise KeyError(f"Key {key} not found in settings.")

def y_branch_2D_topo_init(fdtd):
    fdtd.switchtolayout()
    fdtd.selectall()
    fdtd.delete()

    opt_size_x = settings['design_region_x']
    opt_size_y = settings['design_region_y']
    initial_guess = settings['initial_guess']
    mesh = settings['mesh']
    
    ## SIM PARAMS
    opt_size_x = 3.5e-6
    opt_size_y = 3.5e-6

    size_x = opt_size_x + 0.6e-6
    size_y = opt_size_y + 1e-6

    out_wg_dist = 1.25e-6
    wg_width = 0.5e-6
    mode_width = 3 * wg_width

    wg_index = 2.8
    bg_index = 1.44

    dx = 20e-9

    ## GEOMETRY

    # INPUT WAVEGUIDE
    fdtd.addrect()
    fdtd.set('name', 'input wg')
    fdtd.set('x min', -size_x)
    fdtd.set('x max', -opt_size_x / 2 + 1e-7)
    fdtd.set('y', 0)
    fdtd.set('y span', wg_width)
    fdtd.set('z', 0)
    fdtd.set('z span', 220e-9)
    fdtd.set('index', wg_index)

    ## OUTPUT WAVEGUIDES
    fdtd.addrect()
    fdtd.set('name', 'output wg top')
    fdtd.set('x min', opt_size_x / 2 - 1e-7)
    fdtd.set('x max', size_x)
    fdtd.set('y', out_wg_dist)
    fdtd.set('y span', wg_width)
    fdtd.set('z', 0)
    fdtd.set('z span', 220e-9)
    fdtd.set('index', wg_index)

    fdtd.addrect()
    fdtd.set('name', 'output wg bottom')
    fdtd.set('x min', opt_size_x / 2 - 1e-7)
    fdtd.set('x max', size_x)
    fdtd.set('y', -out_wg_dist)
    fdtd.set('y span', wg_width)
    fdtd.set('z', 0)
    fdtd.set('z span', 220e-9)
    fdtd.set('index', wg_index)

    ## SOURCE
    fdtd.addmode()
    fdtd.set('direction', 'Forward')
    fdtd.set('injection axis', 'x-axis')
    fdtd.set('x', -size_x / 2 + 1e-7)
    fdtd.set('y', 0)
    fdtd.set('y span', mode_width)
    fdtd.set('z', 0)
    fdtd.set('z span', 1e-6)
    fdtd.set('center wavelength', 1550e-9)
    fdtd.set('wavelength span', 0)
    fdtd.set('mode selection', 'fundamental TE mode')

    ## FDTD
    fdtd.addfdtd()
    fdtd.set('dimension', '2D')
    fdtd.set('background index', bg_index)
    fdtd.set('mesh accuracy', mesh)
    fdtd.set('x min', -size_x / 2)
    fdtd.set('x max', size_x / 2)
    fdtd.set('y min', -size_y / 2)
    fdtd.set('y max', size_y / 2)
    fdtd.set('y min bc', 'anti-symmetric')
    fdtd.set('auto shutoff min', 1e-7)

    ## OPTIMIZATION FIELDS MONITOR IN OPTIMIZABLE REGION
    fdtd.addpower()
    fdtd.set('name', 'opt_fields')
    fdtd.set('monitor type', '2D Z-normal')
    fdtd.set('x', 0)
    fdtd.set('x span', opt_size_x)
    fdtd.set('y min', 0)
    fdtd.set('y max', opt_size_y / 2)

    ## FOM FIELDS
    fdtd.addpower()
    fdtd.set('name', 'fom')
    fdtd.set('monitor type', '2D X-normal')
    fdtd.set('x', size_x / 2 - 1e-7)
    fdtd.set('y', out_wg_dist)
    fdtd.set('y span', mode_width)

    fdtd.addmesh()
    fdtd.set('name', 'fom_mesh')
    fdtd.set('override x mesh', True)
    fdtd.set('dx', dx)
    fdtd.set('override y mesh', False)
    fdtd.set('override z mesh', False)
    fdtd.set('x', size_x / 2 - 1e-7)
    fdtd.set('x span', 2 * dx)
    fdtd.set('y', out_wg_dist)
    fdtd.set('y span', mode_width)

    ## For visualization later
    fdtd.addindex()
    fdtd.set('name', 'global_index')
    fdtd.set('x min', -size_x / 2)
    fdtd.set('x max', size_x / 2)
    fdtd.set('y min', -size_y / 2)
    fdtd.set('y max', size_y / 2)

    ## Initial guess
    if initial_guess:
        fdtd.addstructuregroup()
        fdtd.set("name", "initial_guess")
        fdtd.addwaveguide()
        fdtd.set("base width", 500e-9)
        fdtd.set("base height", 500e-9)
        fdtd.set("base angle", 90)
        poles = [
            [-opt_size_x / 2, 0],
            [0, 0],
            [0, out_wg_dist],
            [opt_size_x / 2, out_wg_dist],
        ]
        fdtd.set("poles", poles)
        fdtd.set("index", wg_index)
        fdtd.addtogroup("initial_guess")
        

def y_branch_3D_topo_init(fdtd):
    # Clear existing simulation
    fdtd.switchtolayout()
    fdtd.selectall()
    fdtd.delete()

    opt_size_x = settings['opt_size_x']
    opt_size_y = settings['opt_size_y']
    initial_guess = settings['initial_guess']
    mesh = settings['mesh']

    ## SIM PARAMS
    opt_size_x = 3.5e-6
    opt_size_y = 3.5e-6
    opt_size_z = 0.22e-6

    size_x = opt_size_x + 0.6e-6
    size_y = opt_size_y + 1e-6
    size_z = 1.2e-6

    out_wg_dist = 1.25e-6
    wg_width = 0.5e-6
    mode_width = 3 * wg_width

    wg_index = 3.48
    bg_index = 1.44

    dx = 20e-9

    ## GEOMETRY

    # INPUT WAVEGUIDE
    fdtd.addrect()
    fdtd.set('name', 'input wg')
    fdtd.set('x min', -size_x)
    fdtd.set('x max', -opt_size_x / 2 + 1e-7)
    fdtd.set('y', 0)
    fdtd.set('y span', wg_width)
    fdtd.set('z', 0)
    fdtd.set('z span', 220e-9)
    fdtd.set('index', wg_index)

    ## OUTPUT WAVEGUIDES
    fdtd.addrect()
    fdtd.set('name', 'output wg top')
    fdtd.set('x min', opt_size_x / 2 - 1e-7)
    fdtd.set('x max', size_x)
    fdtd.set('y', out_wg_dist)
    fdtd.set('y span', wg_width)
    fdtd.set('z', 0)
    fdtd.set('z span', 220e-9)
    fdtd.set('index', wg_index)

    fdtd.addrect()
    fdtd.set('name', 'output wg bottom')
    fdtd.set('x min', opt_size_x / 2 - 1e-7)
    fdtd.set('x max', size_x)
    fdtd.set('y', -out_wg_dist)
    fdtd.set('y span', wg_width)
    fdtd.set('z', 0)
    fdtd.set('z span', 220e-9)
    fdtd.set('index', wg_index)

    ## SOURCE
    fdtd.addmode()
    fdtd.set('direction', 'Forward')
    fdtd.set('injection axis', 'x-axis')
    fdtd.set('x', -size_x / 2 + 1e-7)
    fdtd.set('y', 0)
    fdtd.set('y span', mode_width)
    fdtd.set('z', 0)
    fdtd.set('z span', 1e-6)
    fdtd.set('center wavelength', 1550e-9)
    fdtd.set('wavelength span', 0)
    fdtd.set('mode selection', 'fundamental TE mode')

    ## FDTD
    fdtd.addfdtd()
    fdtd.set('dimension', '3D')
    fdtd.set('background index', bg_index)
    fdtd.set('mesh accuracy', mesh)  # Increase this only if the optimization mesh is refined below 20nm
    fdtd.set('x min', -size_x / 2)
    fdtd.set('x max', size_x / 2)
    fdtd.set('y min', -size_y / 2)
    fdtd.set('y max', size_y / 2)
    fdtd.set('z min', -size_z / 2)
    fdtd.set('z max', size_z / 2)
    fdtd.set('force symmetric z mesh', 1)
    fdtd.set('y min bc', 'Anti-Symmetric')
    fdtd.set('auto shutoff min', 1e-7)

    ## OPTIMIZATION FIELDS MONITOR IN OPTIMIZABLE REGION
    fdtd.addpower()
    fdtd.set('name', 'opt_fields')
    fdtd.set('monitor type', '3D')
    fdtd.set('x', 0)
    fdtd.set('x span', opt_size_x)
    fdtd.set('y min', 0)
    fdtd.set('y max', opt_size_y / 2)
    fdtd.set('z', 0)
    fdtd.set('z span', opt_size_z)

    ## FOM FIELDS
    fdtd.addpower()
    fdtd.set('name', 'fom')
    fdtd.set('monitor type', '2D X-normal')
    fdtd.set('x', size_x / 2 - 1e-7)
    fdtd.set('y', out_wg_dist)
    fdtd.set('y span', mode_width)
    fdtd.set('z min', -size_z / 2)
    fdtd.set('z max', size_z / 2)

    fdtd.addmesh()
    fdtd.set('name', 'fom_mesh')
    fdtd.set('override x mesh', True)
    fdtd.set('dx', dx)
    fdtd.set('override y mesh', False)
    fdtd.set('override z mesh', False)
    fdtd.set('x', size_x / 2 - 1e-7)
    fdtd.set('x span', 2 * dx)
    fdtd.set('y', out_wg_dist)
    fdtd.set('y span', mode_width)
    fdtd.set('z min', -size_z / 2)
    fdtd.set('z max', size_z / 2)

    ## For visualization later
    fdtd.addindex()
    fdtd.set('name', 'global_index')
    fdtd.set('monitor type', '3D')
    fdtd.set('x min', -size_x / 2)
    fdtd.set('x max', size_x / 2)
    fdtd.set('y min', -size_y / 2)
    fdtd.set('y max', size_y / 2)
    fdtd.set('z min', -size_z / 2)
    fdtd.set('z max', size_z / 2)
    fdtd.set('enabled', False)

    ## Initial guess
    if initial_guess:
        fdtd.addstructuregroup()
        fdtd.set("name", "initial_guess")
        fdtd.addwaveguide()
        fdtd.set("base width", 500e-9)
        fdtd.set("base height", 220e-9)
        fdtd.set("base angle", 90)
        poles = [
            [-opt_size_x / 2, 0],
            [0, 0],
            [0, 1.25e-6],
            [opt_size_x / 2, 1.25e-6],
        ]
        fdtd.set("poles", poles)
        fdtd.set("index", wg_index)
        fdtd.addtogroup("initial_guess")