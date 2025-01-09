"""
taper.py

This module contains taper GEOMETRIES to be imported and used as polygons or geometries in the optimization object
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

def taper_geo(dimension = 2, type = 'spline', init_params = None, base_script = True, lithography_model = None, **kwargs):
    """
    Generates a taper polygon/geometry defined by either spline interpolation or bezier curves.

    Parameters:
        dimension (int): 2 uses varFDTD, 3 uses FDTD (default)
        type (str): supported types - 'spline' or 'bezier'.
            - 'spline': Uses cubic spline interpolation to generate the taper. This method fits a cubic spline to the control points.
            - 'bezier': Uses Bezier curves to generate the taper. The number of parameters are the bezier curve degrees.
        init_params (np.array): initial parameters for the optimization
        base_script (bool): whether to return a modified base_script in addition to the geometry. 
        lithography_model (str): the lithography model to use with the polygon/geometry.
        
        **kwargs:
            - 'start_x'/'stop_x' (float): Boundaries for x-coordinates of points. 
            - 'start_y'/'stop_y' (float): Boundaries for y-coordinates of points. 
                                          The input and output waveguide widths are based on start_y and stop_y respectively
            - 'num_params' (int): Number of intermediate spline parameters (not including fixed end points)
            - 'min_bound'/'max_bound' (float): Clipping bounds for y-values of the spline.
            - 'poly_wg' (bool): option to include waveguides as a polygon or within the base script
                                if using GDS lithography models, this will default to true. else false.
    Returns:
        FunctionDefinedPolygon_GL/Geometry: A polygon is returned if kwargs poly_wg is not True
                                            Otherwise, a geometry object is returned with waveguides added
        base_script: modified base_script based on the kwargs provided
    """
    # initialize kwargs
    start_x = kwargs.get('start_x', -1.0e-6)
    stop_x = kwargs.get('stop_x', 1e-6)
    start_y = kwargs.get('start_y', 0.25e-6)
    stop_y = kwargs.get('stop_y', 0.6e-6)
    poly_wg = kwargs.get('poly_wg', False)
    max_bound = kwargs.get('max_bound', 0.8e-6)
    min_bound = kwargs.get('min_bound', 0.2e-6)
    num_params = kwargs.get('num_params', 2 if type == 'bezier' else 8) # cubic bezier or 10 point spline. 
    if lithography_model in ['DUV', 'NanoSOI'] and poly_wg is None: # if poly_wg isnt explicitly set to false, set to true
        poly_wg = True

    # Define taper polygon
    eps_in = Material(name = 'Si: non-dispersive')
    eps_out = Material(name = 'SiO2: non-dispersive')
    if type == 'bezier':
        polygon = _create_taper_bezier(start_x, stop_x, start_y, stop_y, num_params, init_params, eps_in, eps_out, min_bound, max_bound)
    else:
        polygon = _create_taper_spline(start_x, stop_x, start_y, stop_y, num_params, init_params, eps_in, eps_out, min_bound, max_bound)
    
    # Define input and output waveguides as polygons, if necessary
    if poly_wg == True:
        polygon_list = [polygon]
        polygon_list = _create_waveguides(polygon_list, start_x, stop_x, start_y, stop_y, eps_in, eps_out)
        polygon = Geometry_GL(polygon_list, 'add')
    
    # add the lithography model
    if lithography_model is not None:
        polygon.lithography_model = LithographyOptimizer(model = lithography_model) # adds it to either the polygon or the geometry
    
    # if no base_script is required, then return polygon
    if base_script == False:
        return polygon 
    
    # else, create a dynamic base script based on kwargs
    if dimension  == 2:
        from base_scripts.varFDTD_taper import taper_init_, update_settings
        for key, value in kwargs.items():
            update_settings(key, value)
        base_script = taper_init_
    else:
        from base_scripts.FDTD_taper import taper_3D_init_, update_settings
        for key, value in kwargs.items():
            update_settings(key, value)
        base_script = taper_3D_init_
        
    return polygon, base_script

def taper_base_script(dimension = 2, **kwargs):
    """
    Returns just the taper basescript based on the kwargs. Useful for combined geometries.

    Parameters:
        dimension (int): 2 sets up a varFDTD simulation, 3 for FDTD (default)
        
        **kwargs:
            - start_x (float): Boundaries for x-coordinates of points.
            - 'start_y'/'stop_y' (float): Boundaries for y-coordinates of points.
            - 'min_bound'/'max_bound' (float): Clipping bounds for y-values of the spline.
            - 'poly_wg' (bool): option to create static input and output waveguides or not.
    Returns:
        base_script: modified base_script based on the kwargs provided
    """
    # else, create a dynamic base script based on kwargs
    if dimension  == 2:
        from base_scripts.varFDTD_taper import taper_init_, update_settings
        for key, value in kwargs.items():
            update_settings(key, value)
        base_script = taper_init_
    else:
        from base_scripts.FDTD_taper import taper_3D_init_, update_settings
        for key, value in kwargs.items():
            update_settings(key, value)
        base_script = taper_3D_init_
        
    return base_script

def _create_taper_spline(start_x, stop_x, start_y, stop_y, num_params, init_params, eps_in, eps_out, min_bound, max_bound):
    initial_points_x = np.linspace(start_x, stop_x, num_params + 2)
    initial_points_y = np.linspace(start_y, stop_y, num_params + 2)
    
    initial_params_y = init_params if init_params is not None else initial_points_y[1:-1] # remove the start and end points, as these will not be varied
    bounds = [(min_bound, max_bound)] * len(initial_params_y)
    
    def spline_taper(params = initial_params_y):
        ''' Defines a taper where the paramaters are the y coordinates of the nodes of a cubic spline. '''
        points_x = np.concatenate(([initial_points_x.min()], initial_points_x[1:-1], [initial_points_x.max()]))
        points_y = np.concatenate(([initial_points_y[0]], params, [initial_points_y[-1]]))
        
        polygon_points_x = np.linspace(min(points_x), max(points_x), 100)
        interpolator = sp.interpolate.interp1d(points_x, points_y, kind = 'cubic')
        polygon_points_y = interpolator(polygon_points_x)
        polygon_points_y = np.clip(polygon_points_y, min_bound, max_bound)
        
        polygon_points_up = [(x, y) for x, y in zip(polygon_points_x, polygon_points_y)]
        polygon_points_down = [(x, -y) for x, y in zip(polygon_points_x, polygon_points_y)]
        polygon_points = np.array(polygon_points_up[::-1] + polygon_points_down)
        return polygon_points

    depth = 220.0e-9
    polygon = FunctionDefinedPolygon_GL(func = spline_taper, 
                                    initial_params = initial_params_y,
                                    bounds = bounds,
                                    z = 0.0,
                                    depth = depth,
                                    eps_out = eps_out, 
                                    eps_in = eps_in,
                                    dx = 2e-9)
    return polygon


def _create_taper_bezier(start_x, stop_x, start_y, stop_y, num_params, init_params, eps_in, eps_out, min_bound, max_bound):
    initial_points_x = np.linspace(start_x, stop_x, num_params + 2)
    initial_points_y = np.linspace(start_y, stop_y, num_params + 2)
    
    initial_params_y = init_params if init_params is not None else initial_points_y[1:-1] # remove the start and end points, as these will not be varied
    bounds = [(min_bound, max_bound)] * len(initial_params_y)
    
    def bernstein_poly(i, n, t):
        return comb(n, i) * (t**i) * ((1 - t)**(n - i))

    def bezier_curve(points, num_points=100):
        n = len(points) - 1
        t = np.linspace(0, 1, num_points)
        curve = np.zeros((num_points, 2))
        for i in range(n + 1):
            curve += np.outer(bernstein_poly(i, n, t), points[i])
        return curve
    
    def bezier_taper(params = initial_params_y):
        '''defines a taper by fitting a Bezier curve to the boundary'''
        points = np.concatenate((
            [[initial_points_x[0], initial_points_y[0]]],
            np.column_stack((initial_points_x[1:-1], params)),
            [[initial_points_x[-1], initial_points_y[-1]]]
        ))
        bezier_points = bezier_curve(points)
        bezier_points[:, 1] = np.clip(bezier_points[:, 1], min_bound, max_bound)
        
        polygon_points_up = [(x, y) for x, y in bezier_points]
        polygon_points_down = [(x, -y) for x, y in bezier_points]
        polygon_points = np.array(polygon_points_up[::-1] + polygon_points_down)
        return polygon_points

    depth = 220.0e-9
    polygon = FunctionDefinedPolygon_GL(func = bezier_taper,
                                    initial_params = initial_params_y,
                                    bounds = bounds,
                                    z = 0.0,
                                    depth = depth,
                                    eps_out = eps_out, 
                                    eps_in = eps_in,
                                    dx = 2e-9)
    return polygon

def _create_waveguides(polygon_list, start_x, stop_x, start_y, stop_y, eps_in, eps_out):
    """Creates input and output waveguides and appends them to the polygons list."""
    def input_wg(params = None):
        return np.array([(start_x-2e-6, start_y), (start_x-2e-6, -start_y), (start_x, -start_y), (start_x, start_y)])
    def output_wg(params=None):
        return np.array([(stop_x + 2e-6, stop_y), (stop_x, stop_y), (stop_x, -stop_y), (stop_x + 2e-6, -stop_y)])
    
    for func in [input_wg, output_wg]:
        polygon_list.append(FunctionDefinedPolygon_GL(
            func=func,
            initial_params=np.empty(0),
            bounds=np.empty((0, 2)),
            z=0.0,
            depth=220e-9,
            eps_out=eps_out,  # Use the variable passed to the function
            eps_in=eps_in,
        ))
        
    return polygon_list