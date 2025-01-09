"""
SWG_finger.py

An SWG finger structure are SWGs attached to the edges of waveguides or slotted waveguides to support modulation
This module contains geometries to be imported and used with an optimization object.
Each geometry supports additional arguments to reconstruct the base_script based on the arguments. 
"""

import numpy as np
import scipy as sp
import lumapi
from lumopt.utilities.materials import Material
from lumopt.geometries.polygon import FunctionDefinedPolygon
from lumopt_gl.geometries.geometry_gl import Geometry_GL
from lumopt_gl.geometries.polygon_gl import FunctionDefinedPolygon_GL
from lumopt_gl.utilities.lithography import LithographyOptimizer
from scipy.special import comb

def SWG_finger_geo(dimension = 2, init_params = None, width_as_param = False, base_script = True, lithography_model = None, **kwargs):
    """
    Generates an strip to SWG finger waveguide taper and the corresponding base script based on the arguments

    Parameters:
        dimension (int): 2 uses varFDTD, 3 uses FDTD (default)
        init_params (np.array): initial parameters for the optimization. params are either the heights or the
                                heights and widths. Each individual grating is parameterized separately.
        width_as_param (bool):  determine whether to use the grating widths as params or only the heights
        base_script (bool): whether to return a modified base_script in addition to the geometry. 
        lithography_model (str): the lithography model to use with the polygon/geometry.
        
        **kwargs:
            - wg_width (float): width of the central waveguide
            - period (float): period of the SWG gratings
            - num_gratings (int): total number of PARAMETERIZED gratings. There will be more determined by...
            - static_gratings (int): generates additional gratings at the end of the parameterized ones
            - swg_width (float): width of each individual grating (or final width if width is a parameter))
            - final_height (float): height of static gratings
            - 'min_bound'/'max_bound' (float): bounds for the SWG grating heights.
            - mesh (int): mesh accuracy. Default 4 for varFDTD, 3 for FDTD.
    Returns:
        FunctionDefinedPolygon_GL/Geometry: A polygon is returned if kwargs poly_wg is not True
                                            Otherwise, a geometry object is returned with waveguides added
        base_script: modified base_script based on the kwargs provided
    """
    # initialize kwargs
    wg_width = kwargs.get('wg_width', 0.5e-6)
    period = kwargs.get('period', 0.24e-6)
    num_gratings = kwargs.get('num_gratings', 20)
    static_gratings = kwargs.get('static_gratings', 5)
    swg_width = kwargs.get('swg_width', 0.12e-6)
    max_bound = kwargs.get('max_bound', 0.7e-6)
    min_bound = kwargs.get('min_bound', 0.07e-6)
    final_height = kwargs.get('min_bound', 0.5e-6)

    # Define taper polygon
    eps_in = Material(name = 'Si: non-dispersive')
    eps_out = Material(name = 'SiO2: non-dispersive')
    polygon = _create_SWG_Fingers(wg_width, period, num_gratings, static_gratings, swg_width, init_params, final_height, eps_in, eps_out, min_bound, max_bound)
    
    # add the lithography model
    if lithography_model is not None:
        polygon.lithography_model=LithographyOptimizer(lithography_model) # adds it to either the polygon or the geometry
    
    # if no base_script is required, then return polygon
    if base_script == False:
        return polygon 
    
    # else, create a dynamic base script based on kwargs
    if dimension  == 2:
        from base_scripts.varFDTD_SWG_fingers import SWG_fingers_init_, update_settings
        for key, value in kwargs.items():
            update_settings(key, value)
        base_script = SWG_fingers_init_
    else:
        from base_scripts.FDTD_SWG_fingers import SWG_fingers_3D_init_, update_settings
        for key, value in kwargs.items():
            update_settings(key, value)
        base_script = SWG_fingers_3D_init_
        
    return polygon, base_script

def _create_SWG_Fingers(wg_width, period, num_gratings, static_gratings, swg_width, init_params, final_height, eps_in, eps_out, min_bound, max_bound):
    """Creates the SWG Fingers polygons and adds them to a geometry."""
    
    if init_params is None:
        init_params = np.linspace(final_height, min_bound, num_gratings)

    polygons = []
    center_width = period - swg_width
    length = (num_gratings + static_gratings) * period + center_width # this is the length from one end
    bounds = [(min_bound,max_bound)]
    
    # create waveguide
    def center_waveguide(params):
        """Defines the geometry of the central waveguide"""
        x1 = -length - 3e-6
        x2 = length + 3e-6
        y1 = -wg_width * 0.5
        y2 = wg_width * 0.5
        return np.array([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])
    
    waveguide = FunctionDefinedPolygon(func=center_waveguide, initial_params=np.empty(0), bounds=np.empty((0, 2)), z=0,
                                        depth=220e-9, eps_out=eps_out, eps_in=eps_in)
    polygons.append(waveguide)
    
    def grating_factory(i):
        """Factory function to create grating functions for both static and parameterized grating geometries based on index."""
        x1 = center_width/2 + i * period
        x2 = x1 + swg_width
        y1 = wg_width * 0.5 + final_height
        y2 = -wg_width * 0.5 - final_height
        if i < static_gratings:
            def grating_func(params = None):
                return np.array([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])
            def symmetrical_grating_func(params = None):
                return np.array([(-x2, y2), (-x2, y1), (-x1, y1), (-x1, y2)])
        else:
            def grating_func(params):
                y1 = wg_width * 0.5 + params[0]
                y2 = -wg_width * 0.5 - params[0]
                return np.array([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])
            def symmetrical_grating_func(params):
                y1 = wg_width * 0.5 + params[0]
                y2 = -wg_width * 0.5 - params[0]
                return np.array([(-x2, y2), (-x2, y1), (-x1, y1), (-x1, y2)])
        return grating_func, symmetrical_grating_func
    
    for i in range(static_gratings):
        grating_func, symmetrical_grating_func = grating_factory(i)
        grating = FunctionDefinedPolygon(func=grating_func, initial_params=np.empty(0), bounds=np.empty((0, 2)), z=0,
                                         depth=220e-9, eps_out=eps_out, eps_in=eps_in)
        symmetrical_grating = FunctionDefinedPolygon(func=symmetrical_grating_func, initial_params=np.empty(0), bounds=np.empty((0, 2)), z=0,
                                                     depth=220e-9, eps_out=eps_out, eps_in=eps_in)
        polygons.extend([grating, symmetrical_grating])

    # Handle dynamic gratings
    for i in range(num_gratings):
        grating_func, symmetrical_grating_func = grating_factory(i + static_gratings)
        grating = FunctionDefinedPolygon(func=grating_func, initial_params=init_params[i:i + 1], bounds=bounds, z=0,
                                         depth=220e-9, eps_out=eps_out, eps_in=eps_in, edge_precision=5, dx=1e-8)
        symmetrical_grating = FunctionDefinedPolygon(func=symmetrical_grating_func, initial_params=init_params[i:i + 1], bounds=bounds,
                                                     z=0, depth=220e-9, eps_out=eps_out, eps_in=eps_in, edge_precision=5, dx=1e-8)
        symmetrical_geo = Geometry_GL([grating, symmetrical_grating], 'mul') # mul because they share the same parameter!
        polygons.append(symmetrical_geo)

    geo = Geometry_GL(polygons, 'add')
    
    return geo
