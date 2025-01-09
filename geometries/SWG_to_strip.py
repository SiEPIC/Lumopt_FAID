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

def SWG_to_strip(dimension = 2, init_params = None, base_script = True, lithography_model = None, **kwargs):
    """
    Generates an strip to SWG converter and the corresponding base script based on the arguments

    Parameters:
        dimension (int): 2 uses varFDTD, 3 uses FDTD (default)
        init_params (np.array): initial parameters for the optimization. params are either the heights or the
                                heights and widths. Each individual grating is parameterized separately.
                                Optional. Defaults to none. 
                                params for gratings start at the center and outwards.
                                [param1height, param1width, param2height, param2width, etc.]
                                The last few params are reserved for waveguide bridge parameters and duty cycle
        base_script (bool): whether to return a modified base_script in addition to the geometry. Default True.
        lithography_model (str): the lithography model to use with the polygon/geometry. Default None.
        
        **kwargs:
            - wg_width (float): width of the central waveguide
            - bridge_length (float): the PERCENTAGE of the converter length that is taken up by the waveguide bridge
            - period (float): period of the SWG gratings.
            - num_gratings (int): total number of PARAMETERIZED gratings. There will be more determined by...
            - static_gratings (int): generates additional gratings at the end of the parameterized ones
            - swg_width (float): width of each individual grating (or final width if width is a parameter))
            - final_height (float): height of static gratings
            - 'SWG_min_bound'/'SWG_max_bound' (float): bounds for the SWG grating heights.
            - bridge_min_bound (float): the minimum width of the bridge
            - mesh (int): mesh accuracy. Default 4 for varFDTD, 3 for FDTD.
            - width_as_param (bool). Whether to parameterize the widths of individual gratings. Defaults to false. 
            - parameterize_duty_cycle (int): Whether to parameterize the duty cycle. Defaults to false.
                                             If true, then widths are not parameterized individually. 
    Returns:
        FunctionDefinedPolygon_GL/Geometry: A polygon is returned if kwargs poly_wg is not True
                                            Otherwise, a geometry object is returned with waveguides added
        base_script: modified base_script based on the kwargs provided
    """
    
    grating_params = {
        'period': kwargs.get('period', 0.24e-6),
        'num_gratings': kwargs.get('num_gratings', 20),
        'static_gratings': kwargs.get('static_gratings', 7),
        'swg_width': kwargs.get('swg_width', 0.12e-6),
        'SWG_max_bound': kwargs.get('SWG_max_bound', 0.4e-6),
        'SWG_min_bound': kwargs.get('SWG_min_bound', 0.05e-6),
        'final_height': kwargs.get('final_height', 0.25e-6),
    }
    
    center_width = grating_params['period'] - grating_params['swg_width']
    length = (grating_params['num_gratings'] + grating_params['static_gratings']) * grating_params['period'] + center_width
    
    other_geo_params = {
        'wg_width': kwargs.get('wg_width', 0.35e-6),
        'bridge_length': kwargs.get('bridge_length', 0.75),
        'bridge_min_bound': kwargs.get('bridge_min_bound', 0.07e-6),
        }
    
    eps_in = Material(name = 'Si: non-dispersive')
    eps_out = Material(name = 'SiO2: non-dispersive')

    
    width_as_param = kwargs.get('width_as_param', False)
    parameterize_duty_cycle = kwargs.get('parameterize_duty_cycle', False)
    
    waveguides = _create_waveguides(other_geo_params['wg_width'], length, eps_in, eps_out)
    SWG_gratings = _create_SWG_Fingers(grating_params, other_geo_params, center_width,init_params, eps_in, eps_out)
    waveguide_bridge = _create_waveguide_bridge(grating_params, other_geo_params, init_params, length, eps_in, eps_out)
    
    geo = Geometry_GL([waveguides, SWG_gratings, waveguide_bridge], 'add')
    # add the lithography model
    if lithography_model is not None:
        geo.lithography_model=LithographyOptimizer(lithography_model) # adds it to either the polygon or the geometry
    
    # if no base_script is required, then return polygon
    if base_script == False:
        return geo 
    
    # else, create a dynamic base script based on kwargs
    if dimension  == 2:
        from base_scripts.varFDTD_SWG_to_strip import SWG_to_strip_init_, update_settings
        for key, value in kwargs.items():
            update_settings(key, value)
        base_script = SWG_to_strip_init_
    else:
        from base_scripts.FDTD_SWG_to_strip import SWG_to_strip_3D_init_, update_settings
        for key, value in kwargs.items():
            update_settings(key, value)
        base_script = SWG_to_strip_3D_init_
        
    return geo, base_script

def _create_waveguides(wg_width, length, eps_in, eps_out):
    """Creates the input and output waveguides""" 
    
    def input_waveguide(params):
        """Defines the geometry of the central waveguide"""
        x1 = -length
        x2 = -length - 3e-6
        y1 = -wg_width * 0.5
        y2 = wg_width * 0.5
        return np.array([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])
    
    def output_waveguide(params):
        """Defines the geometry of the central waveguide"""
        x1 = length
        x2 = length + 3e-6
        y1 = -wg_width * 0.5
        y2 = wg_width * 0.5
        return np.array([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])
    
    input_wg = FunctionDefinedPolygon(func=input_waveguide, initial_params=np.empty(0), bounds=np.empty((0, 2)), z=0,
                                        depth=220e-9, eps_out=eps_out, eps_in=eps_in)
    output_wg = FunctionDefinedPolygon(func=output_waveguide, initial_params=np.empty(0), bounds=np.empty((0, 2)), z=0,
                                        depth=220e-9, eps_out=eps_out, eps_in=eps_in)
    
    waveguide_geo = Geometry_GL([input_wg,output_wg], 'add')
    
    return waveguide_geo
    
def _create_SWG_Fingers(grating_params_dict, geo_params_dict, center_width,init_params, eps_in, eps_out):
    """Creates the SWG Fingers polygons and adds them to a geometry."""
    
    period = grating_params_dict['period']
    num_gratings = grating_params_dict['num_gratings']
    static_gratings = grating_params_dict['static_gratings']
    swg_width = grating_params_dict['swg_width']
    SWG_max_bound = grating_params_dict['SWG_max_bound']
    SWG_min_bound = grating_params_dict['SWG_min_bound']
    final_height = grating_params_dict['final_height']
    wg_width = geo_params_dict['wg_width']
    
    init_heights = np.linspace(wg_width/2, wg_width/2, num=num_gratings) if not init_params else init_params[:num_gratings]
    
    polygons = []
    bounds = [(SWG_min_bound,SWG_max_bound)]
    
    def grating_factory(i):
        """Factory function to create grating functions for both static and parameterized grating geometries based on index."""
        x1 = center_width/2 + i * period
        x2 = x1 + swg_width
        y1 = final_height
        y2 = -final_height
        if i < static_gratings:
            def grating_func(params):
                return np.array([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])
            def symmetrical_grating_func(params):
                return np.array([(-x2, y2), (-x2, y1), (-x1, y1), (-x1, y2)])
        else:
            def grating_func(params):
                y1 = params[0]
                y2 = -params[0]
                return np.array([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])
            def symmetrical_grating_func(params):
                y1 = params[0]
                y2 = -params[0]
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
        grating = FunctionDefinedPolygon(func=grating_func, initial_params=init_heights[i:i + 1], bounds=bounds, z=0,
                                         depth=220e-9, eps_out=eps_out, eps_in=eps_in, edge_precision=5, dx=2e-8)
        symmetrical_grating = FunctionDefinedPolygon(func=symmetrical_grating_func, initial_params=init_heights[i:i + 1], bounds=bounds,
                                                     z=0, depth=220e-9, eps_out=eps_out, eps_in=eps_in, edge_precision=5, dx=2e-8)
        
        symmetrical_geo = Geometry_GL([grating, symmetrical_grating], 'mul') # mul because they share the same parameter!
        
        polygons.append(symmetrical_geo)

    geo = Geometry_GL(polygons, 'add')
    
    return geo

def _create_waveguide_bridge(grating_params_dict, geo_params_dict, init_params, length, eps_in, eps_out):
    
    num_gratings = grating_params_dict['num_gratings']
    SWG_max_bound = grating_params_dict['SWG_max_bound']
    wg_width = geo_params_dict['wg_width']
    bridge_length = geo_params_dict['bridge_length']
    bridge_min_bound = geo_params_dict['bridge_min_bound']
    
    bridge_points = len(init_params) - num_gratings if init_params is not None else 6
    initial_points_x = np.linspace(length * (1-bridge_length), length, bridge_points)
    init_points_y = np.linspace(bridge_min_bound, wg_width/2, bridge_points)
    init_params_y = init_points_y[1:-1] if not init_params else init_params[num_gratings:]

    bounds = [(bridge_min_bound, SWG_max_bound)] * init_params_y.size
    
    def bridge(params = init_params_y):
        points_x = np.concatenate(([initial_points_x.min()], initial_points_x[1:-1], [initial_points_x.max()]))
        points_y = np.concatenate(([bridge_min_bound], params, [wg_width/2]))

        polygon_points_x = np.linspace(min(points_x), max(points_x), 100)
        interpolator = sp.interpolate.interp1d(points_x, points_y, kind = 'cubic')
        polygon_points_y = interpolator(polygon_points_x)
        polygon_points_y = np.clip(polygon_points_y, bridge_min_bound, SWG_max_bound)
        
        polygon_points_up = [(x, y) for x, y in zip(polygon_points_x, polygon_points_y)]
        polygon_points_down = [(x, -y) for x, y in zip(polygon_points_x, polygon_points_y)]
        polygon_points = np.array(polygon_points_up[::-1] + polygon_points_down)
        return polygon_points
    
    def symmetrical_bridge(params = init_params_y):
        polygon_points = bridge(params)
        reflected_points = [(-x, y) for x, y in polygon_points]
        # Reverse the list to maintain CCW orientation
        reflected_points.reverse()
        return np.array(reflected_points)
    
    bridge_polygon = FunctionDefinedPolygon_GL(func = bridge, initial_params = init_params_y, bounds = bounds, z = 0.0,
                                    depth = 220e-9, eps_out = eps_out, eps_in = eps_in, dx = 2e-8)
    
    symmetrical_bridge_polygon = FunctionDefinedPolygon_GL(func = symmetrical_bridge, initial_params = init_params_y, bounds = bounds, z = 0.0,
                                    depth = 220e-9, eps_out = eps_out, eps_in = eps_in, dx = 2e-8)
    
    bridge_geo = Geometry_GL([bridge_polygon, symmetrical_bridge_polygon], 'mul') # symmetrical bridges
    
    return bridge_geo
    
def _create_gratings_duty_cycle():
    print('does nothing right now')