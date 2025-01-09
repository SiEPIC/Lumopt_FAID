
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
    'mesh': 3,
}

def update_settings(key, value):
    """Update configuration settings for the Y-branch simulation."""
    global settings
    if key in settings:
        settings[key] = value
    else:
        raise KeyError(f"Key {key} not found in settings.")


    
def SWG_fingers_3D_init_(fdtd): 
           
    ## CLEAR SESSION
    fdtd.switchtolayout()
    fdtd.selectall()
    fdtd.delete()
    
    num_gratings = settings['num_gratings']
    static_gratings = settings['static_gratings']
    period = settings['period']
    wg_width = settings['wg_width']
    mesh_accuracy = settings['mesh']
    max_bound_height = settings['max_bound_height']

    total_gratings = num_gratings + static_gratings
    design_region_x = 2 * total_gratings * period
    design_region_y = max_bound_height * 1.1

    
    ## SIM PARAMS
    size_x = design_region_x + 2e-6  # slightly larger than the design region
    size_y = max_bound_height + 4e-6
    size_z= 2e-6
    mesh_x = 20e-9
    mesh_y = 20e-9
    mesh_z = 20e-9
    finer_mesh_size_x = design_region_x + 0.8e-6
    finer_mesh_size_y = 2 * max_bound_height + 0.2e-6  + wg_width
    finer_mesh_size= size_x * 0.8 
    lam_c = 1.550e-6  # will be overridden from opt if necessary
    
    ## MATERIAL
    opt_material = fdtd.addmaterial('Dielectric')
    fdtd.setmaterial(opt_material, 'name', 'Si: non-dispersive')
    n_opt = fdtd.getindex('Si (Silicon) - Palik', c / lam_c)
    fdtd.setmaterial('Si: non-dispersive', 'Refractive Index', n_opt)
    
    sub_material = fdtd.addmaterial('Dielectric')
    fdtd.setmaterial(sub_material, 'name', 'SiO2: non-dispersive')
    n_sub = fdtd.getindex('SiO2 (Glass) - Palik', c / lam_c)
    fdtd.setmaterial('SiO2: non-dispersive', 'Refractive Index', n_sub)
    fdtd.setmaterial('SiO2: non-dispersive', "color", np.array([0, 0, 0, 0]))
    
    ## GEOMETRY
    
    fdtd.addrect()
    fdtd.set('name', 'sub')
    fdtd.set('x span', 2 * design_region_x + 4e-6) 
    fdtd.set('y span', size_y)
    fdtd.set('z span', 8e-6)
    fdtd.set('y', 0)
    fdtd.set('x', 0)
    fdtd.set('z', 0)
    fdtd.set('material', 'SiO2: non-dispersive')
    fdtd.set('override mesh order from material database', 1)
    fdtd.set('mesh order', 3)
    fdtd.set('alpha', 0.8)
    fdtd.set('mesh order', 3); 
    
    ## FDTD
    fdtd.addfdtd()
    fdtd.set('mesh accuracy', mesh_accuracy)
    fdtd.set('dimension','3D')
    fdtd.set('x min', -size_x / 2)
    fdtd.set('x max', size_x / 2)
    fdtd.set('y min', -size_y / 2)
    fdtd.set('y max', size_y / 2)
    fdtd.set('z min',-size_z/2.0)
    fdtd.set('z max',size_z/2.0)
    fdtd.set('force symmetric y mesh', 1)
    fdtd.set('force symmetric z mesh',1)
    fdtd.set('z min bc','Symmetric')
    fdtd.set('y min bc', 'Anti-Symmetric')
    fdtd.set('z', 0)

    
    ## SOURCE
    fdtd.addmode()
    fdtd.set('direction', 'Forward')
    fdtd.set('injection axis', 'x-axis')
    fdtd.set('y', 0)
    fdtd.set('y span', finer_mesh_size_y)
    fdtd.set('x', -design_region_x*0.5 - 0.3e-6)
    fdtd.set('center wavelength', 1550e-9)
    fdtd.set('wavelength span', 0)
    fdtd.set('mode selection','fundamental TE mode');
    
    
    ## MESH IN OPTIMIZABLE REGION
    fdtd.addmesh()
    fdtd.set('x', 0)
    fdtd.set('x span', finer_mesh_size_x)
    fdtd.set('y', 0)
    fdtd.set('y span', finer_mesh_size_y)
    fdtd.set('z',0)
    fdtd.set('z span', 0.24e-6)
    fdtd.set('dx', mesh_x)
    fdtd.set('dy', mesh_y)
    fdtd.set('dz',mesh_z)
    
    ## OPTIMIZATION FIELDS MONITOR IN OPTIMIZABLE REGION
    
    fdtd.addpower()
    fdtd.set('name', 'opt_fields')
    fdtd.set('monitor type', '3D')
    fdtd.set('x', 0)
    fdtd.set('x span', finer_mesh_size_x)
    fdtd.set('y', 0)
    fdtd.set('y span', finer_mesh_size_y)
    fdtd.set('z', 0)
    fdtd.set('z span', 0.24e-6)
    
    ## FOM FIELDS
    
    fdtd.addpower()
    fdtd.set('name', 'fom')
    fdtd.set('monitor type','2D X-Normal');
    fdtd.set('x', design_region_x * 0.5 + 0.3e-6)
    fdtd.set('y', 0)
    fdtd.set('y span', finer_mesh_size_y)
    fdtd.set('z', 0)
    fdtd.set('z span', size_z)

