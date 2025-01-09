
######## IMPORTS ########
# General purpose imports
import lumapi
import numpy as np
from scipy.constants import c

settings = {
    'start_x': -1e-6,
    'stop_x': 1e-6,
    'start_y': 0.25e-6,
    'stop_y': 0.6e-6,
    'min_bound': 0.2e-6,
    'max_bound': 0.8e-6,
    'num_params': 10,
    'poly_wg': False,
    'mesh': 3,
    }

def update_settings(key, value):
    """Update configuration settings for the Y-branch simulation."""
    global settings
    if key in settings:
        settings[key] = value
    else:
        raise KeyError(f"Key {key} not found in settings.")

def y_branch3D_init_(fdtd): 
	## CLEAR SESSION
    fdtd.switchtolayout()
    fdtd.selectall()
    fdtd.delete()
    
    # Settings Params
    
    x_start = settings['start_x']
    x_end = settings['stop_x']
    y_start = settings['start_y']
    y_end = settings['stop_y']
    min_bounds = settings['min_bound']
    max_bounds = settings['max_bound']
    poly_wg = settings['poly_wg']
    wg_width = 2 * y_start
    mesh_accuracy=settings['mesh']
    
    ## SIM PARAMS
    size_x=(x_end - x_start) * 1.5 # slightly larger than the design region
    size_y= size_x;
    size_z=1.2e-6;
    mesh_x=20e-9;
    mesh_y=20e-9;
    mesh_z=20e-9;
    finer_mesh_size= (x_end - x_start) + 0.4e-6
    finer_mesh_size_z=0.6e-6;
    lam_c = 1.550e-6;
    
    # MATERIAL
    opt_material=fdtd.addmaterial('Dielectric');
    fdtd.setmaterial(opt_material,'name','Si: non-dispersive');
    n_opt = fdtd.getindex('Si (Silicon) - Palik',c/lam_c);
    fdtd.setmaterial('Si: non-dispersive','Refractive Index',n_opt);
    
    sub_material=fdtd.addmaterial('Dielectric');
    fdtd.setmaterial(sub_material,'name','SiO2: non-dispersive');
    n_sub = fdtd.getindex('SiO2 (Glass) - Palik',c/lam_c);
    fdtd.setmaterial('SiO2: non-dispersive','Refractive Index',n_sub);
    fdtd.setmaterial('SiO2: non-dispersive',"color", np.array([0, 0, 0, 0]));
    
    ## GEOMETRY
    
    #INPUT WAVEGUIDE
    
    if not poly_wg:  # if we aren't passing the waveguides as polygons, then they are static.
        
        fdtd.addrect()
        fdtd.set('name', 'input wg')
        fdtd.set('x span', 3e-6)
        fdtd.set('y span', wg_width)
        fdtd.set('z span', 220e-9)
        fdtd.set('y', 0)
        fdtd.set('x', x_start - 1.5e-6)
        fdtd.set('z', 0)
        fdtd.set('material', 'Si: non-dispersive')
        
        # OUTPUT WAVEGUIDES
        
        fdtd.addrect()
        fdtd.set('name', 'output wg top')
        fdtd.set('x span', 3e-6)
        fdtd.set('y span', wg_width)
        fdtd.set('z span', 220e-9)
        fdtd.set('y', y_end - 0.5 * wg_width)
        fdtd.set('x', x_end + 1.5e-6)
        fdtd.set('z', 0)
        fdtd.set('material', 'Si: non-dispersive')
        
        fdtd.addrect()
        fdtd.set('name', 'output wg bottom')
        fdtd.set('x span', 3e-6)
        fdtd.set('y span', wg_width)
        fdtd.set('z span', 220e-9)
        fdtd.set('y', -1 * (y_end - 0.5 * wg_width))
        fdtd.set('x', x_end + 1.5e-6)
        fdtd.set('z', 0)
        fdtd.set('material', 'Si: non-dispersive')

    
    fdtd.addrect();
    fdtd.set('name','sub');
    fdtd.set('x span',2* (x_end + 3e-6));
    fdtd.set('y span',2* (x_end + 3e-6));
    fdtd.set('z span',10e-6);
    fdtd.set('y',0);
    fdtd.set('x',0);
    fdtd.set('z',0);
    fdtd.set('material','SiO2: non-dispersive');
    fdtd.set('override mesh order from material database',1);
    fdtd.set('mesh order',3);
    fdtd.set('alpha',0.8);
    
    ## FDTD
    fdtd.addfdtd();
    fdtd.set('mesh accuracy',mesh_accuracy);
    fdtd.set('dimension','3D');
    fdtd.set('x min',-size_x/2);
    fdtd.set('x max',size_x/2);
    fdtd.set('y min',-size_y/2);
    fdtd.set('y max',size_y/2);
    fdtd.set('z min',-size_z/2.0);
    fdtd.set('z max',size_z/2.0);
    fdtd.set('force symmetric y mesh',1);
    fdtd.set('force symmetric z mesh',1);
    fdtd.set('z min bc','Symmetric');
    fdtd.set('y min bc','Anti-Symmetric');
    
    
    ## SOURCE
    fdtd.addmode();
    fdtd.set('direction','Forward');
    fdtd.set('injection axis','x-axis');
    #fdtd.set('polarization angle',0);
    fdtd.set('y',0);
    fdtd.set("y span",size_y);
    fdtd.set('x', x_start - 0.25e-6);
    fdtd.set('center wavelength',lam_c);
    fdtd.set('wavelength span',0);
    fdtd.set('mode selection','fundamental TE mode');
    
    
    ## MESH IN OPTIMIZABLE REGION
    fdtd.addmesh();
    fdtd.set('x',0);
    fdtd.set('x span',finer_mesh_size);
    fdtd.set('y',0);
    fdtd.set('y span',finer_mesh_size);
    fdtd.set('z',0);
    fdtd.set('z span',0.4e-6);
    fdtd.set('dx',mesh_x);
    fdtd.set('dy',mesh_y);
    fdtd.set('dz',mesh_z);
    
    ## OPTIMIZATION FIELDS MONITOR IN OPTIMIZABLE REGION
    
    fdtd.addpower();
    fdtd.set('name','opt_fields');
    fdtd.set('monitor type','3D');
    fdtd.set('x',0);
    fdtd.set('x span',finer_mesh_size);
    fdtd.set('y',0);
    fdtd.set('y span',finer_mesh_size);
    fdtd.set('z',0);
    fdtd.set('z span',0.4e-6);
    
    ## FOM FIELDS
    
    fdtd.addpower();
    fdtd.set('name','fom');
    fdtd.set('monitor type','2D X-Normal');
    fdtd.set('x',x_end + 0.2e-6);
    fdtd.set('y',0);
    fdtd.set('y span',size_y);
    fdtd.set('z',0);
    fdtd.set('z span',size_z)