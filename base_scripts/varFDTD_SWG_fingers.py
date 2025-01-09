######## IMPORTS ########
# General purpose imports
import lumapi
import numpy as np
from scipy.constants import c

# settings dict allows for convenient adjustment of key parameters at base script initialization
settings = {
    'num_gratings': 20,
    'static_gratings': 5,
    'period': 0.24e-6,
    'min_bound_height': 0.05e-6,
    'max_bound_height': 0.7e-6,
    'wg_width': 0.5e-6,
    'mesh': 4,
}

def update_settings(key, value):
    """Update configuration settings for the Y-branch simulation."""
    global settings
    if key in settings:
        settings[key] = value
    else:
        raise KeyError(f"Key {key} not found in settings.")


    
def SWG_fingers_init_(mode): 
           
    ## CLEAR SESSION
    mode.switchtolayout()
    mode.selectall()
    mode.delete()
    
    num_gratings = settings['num_gratings']
    static_gratings = settings['static_gratings']
    period = settings['period']
    wg_width = settings['wg_width']
    mesh_accuracy = settings['mesh']
    max_bound_height = settings['max_bound_height']

    total_gratings = num_gratings + static_gratings
    design_region_x = 2 * total_gratings * period

    
    ## SIM PARAMS
    size_x = design_region_x + 2e-6  # slightly larger than the design region
    size_y = max_bound_height + 6e-6
    mesh_x = 20e-9
    mesh_y = 20e-9
    finer_mesh_size_x = design_region_x + 0.8e-6
    finer_mesh_size_y = 2 * max_bound_height + 0.4e-6  + wg_width 
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
    
    mode.addrect()
    mode.set('name', 'sub')
    mode.set('x span', 2 * design_region_x + 4e-6) 
    mode.set('y span', size_y)
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
    
    #mode.set('x0', -design_region_x * 0.5 - 0.3e-6)
    #mode.set('number of test points', 4)
    #mode.set('test points', np.array([[0, 0], [design_region_x * 0.5 + 0.3e-6, 0.1e-6], [design_region_x * 0.5 + 0.3e-6, -0.1e-6], [design_region_x * 0.5 + 0.3e-6, 0]]))
    
    ## SOURCE
    mode.addmodesource()
    mode.set('direction', 'Forward')
    mode.set('injection axis', 'x-axis')
    # mode.set('polarization angle',0)
    mode.set('y', 0)
    mode.set('y span', size_y)
    mode.set('x', -design_region_x*0.5 - 0.3e-6)
    mode.set('center wavelength', 1550e-9)
    mode.set('wavelength span', 0)
    mode.set('mode selection', 'fundamental mode')
    
    
    ## MESH IN OPTIMIZABLE REGION
    mode.addmesh()
    mode.set('x', 0)
    mode.set('x span', finer_mesh_size_x)
    mode.set('y', 0)
    mode.set('y span', finer_mesh_size_y)
    mode.set('dx', mesh_x)
    mode.set('dy', mesh_y)
    
    ## OPTIMIZATION FIELDS MONITOR IN OPTIMIZABLE REGION
    
    mode.addpower()
    mode.set('name', 'opt_fields')
    mode.set('monitor type', '2D Z-normal')
    mode.set('x', 0)
    mode.set('x span', finer_mesh_size_x)
    mode.set('y', 0)
    mode.set('y span', finer_mesh_size_y)
    mode.set('z', 0)
    
    ## FOM FIELDS
    
    mode.addpower()
    mode.set('name', 'fom')
    mode.set('monitor type', 'Linear Y')
    mode.set('x', design_region_x * 0.5 + 0.3e-6)
    mode.set('y', 0)
    mode.set('y span', size_y)
    mode.set('z', 0)

