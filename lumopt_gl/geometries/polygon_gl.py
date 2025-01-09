import sys
import numpy as np
import scipy as sp
import random
import lumapi
from lumopt.geometries.geometry import Geometry
from lumopt.utilities.materials import Material
from lumopt.geometries.polygon import Polygon
from lumopt.geometries.polygon import FunctionDefinedPolygon
from lumopt_gl.utilities.edge import Edge

class Polygon_GL(Polygon):
    """ 
        A polygon object that supports lithography transformation and penalty terms for freeform optimization.
        
        New Parameters
        ----------
        :param lithography:     flag to use lithography model specified in the lithography class
    """
    
    def __init__(self, points, z,depth, eps_out, eps_in, edge_precision, deps_num_threads=1):
        super().__init__( points, z, depth, eps_out, eps_in, edge_precision, deps_num_threads)
        self.bounds = [(-float('inf'), float('inf'))] * (points.size)
        
    def make_edges(self):
        '''Creates all the edge objects'''
        edges=[]

        for i,point in enumerate(self.points):
            edges.append(Edge(self.points[i-1],self.points[i],eps_in=self.eps_in,eps_out=self.eps_out,z=self.z,depth=self.depth))
        self.edges=edges
        
    def calculate_gradients(self, gradient_fields):
        ''' We calculate gradients with respect to moving each point in x or y direction '''
        self.make_edges()
        print('Calculating gradients for {} edges'.format(len(self.edges)))
        gradient_pairs_edges=[]
        for edge in self.edges:
            gradient_pairs_edges.append(edge.derivative(gradient_fields, n_points = self.edge_precision))
            sys.stdout.write('.')
        print('')
        #the gradients returned for an edge derivative are the gradients with respect to moving each end point perpendicular to that edge
        #This is not exactly what we are looking for here, since we want the derivative w/ respect to moving each point
        #in the x or y direction, so coming up is a lot of projections...

        gradients = list()
        for i,point in enumerate(self.points):
            deriv_edge_1 = gradient_pairs_edges[i][1]
            normal_edge_1 = self.edges[i].normal
            deriv_edge_2 = gradient_pairs_edges[(i+1)%len(self.edges)][0]
            normal_edge_2 = self.edges[(i+1)%len(self.edges)].normal
            deriv_x = np.dot([1,0,0], np.outer(normal_edge_1, deriv_edge_1).squeeze() + np.outer(normal_edge_2, deriv_edge_2).squeeze())
            deriv_y = np.dot([0,1,0], np.outer(normal_edge_1, deriv_edge_1).squeeze() + np.outer(normal_edge_2, deriv_edge_2).squeeze())
            gradients.append(deriv_x)
            gradients.append(deriv_y)
        self.gradients.append(gradients)
        return self.gradients[-1]
        
class FunctionDefinedPolygon_GL(Polygon_GL):
    """ 
        Constructs a polygon from a user defined function that takes the optimization parameters and returns a set of vertices defining a polygon.
        The polygon vertices returned by the function must be defined as a numpy array of coordinate pairs np.array([(x0,y0),...,(xn,yn)]). THE 
        VERTICES MUST BE ORDERED IN A COUNTER CLOCKWISE DIRECTION.

        Parameters
        ----------
        :param fun:                 function that takes the optimization parameter values and returns a polygon.
        :param initial_params:      initial optimization parameter values.
        :param bounds:              bounding ranges (min/max pairs) for each optimization parameter.
        :param z:                   center of polygon along the z-axis.
        :param depth:               span of polygon along the z-axis.
        :param eps_out:             permittivity of the material around the polygon.
        :param eps_in:              permittivity of the polygon material.
        :param edge_precision:      number of quadrature points along each edge for computing the FOM gradient using the shape derivative approximation method.
        :param dx:                  step size for computing the FOM gradient using permittivity perturbations.
        :param lithography_model:   provide lithography model to polygon directly if using boundary perturbation (use_deps = False). DO NOT USE otherwise.
    """

    def __init__(self, func, initial_params, bounds, z, depth, eps_out, eps_in, edge_precision = 5, dx = 1.0e-10, deps_num_threads = 1, lithography_model = None):
        self.func = func
        if initial_params is None:
            initial_params=np.empty(0)
        if bounds is None:
            bounds=np.empty((0, 2))
        self.current_params = np.array(initial_params).flatten()
        points = func(self.current_params)
        super(FunctionDefinedPolygon_GL, self).__init__(points, z, depth, eps_out, eps_in, edge_precision, deps_num_threads=deps_num_threads)
        self.bounds = bounds
        self.dx = float(dx)
        if self.dx <= 0.0:
            raise UserWarning("step size must be positive.")
        self.params_hist = list(self.current_params)
        
        self.lithography_model = lithography_model
        if lithography_model and lithography_model not in ['Gaussian_DUV', 'ML']:
            raise ValueError("Unsupported lithography model. To use GDS Lithography models please use a Geometry and include input/output waveguides.")

    def update_geometry(self, params, sim = None):
        if self.lithography_model and self.lithography_model in ['Gaussian_DUV']:
            # retrieves the filtered params
            points=self.func(params)
            self.litho_params = self.lithography_model.apply_lithography_boundary(params, points)
            self.points=self.func(self.litho_params)
            self.current_params=self.litho_params
        else:
            self.points=self.func(params)
            self.current_params=params
        self.params_hist.append(params)

    def get_current_params(self):
        return self.current_params

    def calculate_gradients(self, gradient_fields):
        try:
            if self.lithography_model.model in ["GaussianDUV", "GaussianEBL", "ML_Model"]:
                grad = self.lithography_model.apply_jacobian(grad)
        except AttributeError:
            pass
        polygon_gradients = np.array(Polygon.calculate_gradients(self, gradient_fields))
        polygon_points_linear = self.func(self.current_params).reshape(-1)
        gradients = list()
        for i, param in enumerate(self.current_params):
            d_params = np.array(self.current_params.copy())
            d_params[i] += self.dx
            d_polygon_points_linear = self.func(d_params).reshape(-1)
            partial_derivs = (d_polygon_points_linear - polygon_points_linear) / self.dx
            gradients.append(np.dot(partial_derivs, polygon_gradients))
        self.gradients.append(gradients)
        return np.array(self.gradients[-1])

    def add_poly_script(self, sim, points, only_update):
        poly_name = 'polygon_{}'.format(self.hash)
        try:
            sim.fdtd.setnamed(poly_name, 'x', 0.0)
        except:
            sim.fdtd.addpoly()
            sim.fdtd.set('name', poly_name)
        """if not only_update:
            sim.fdtd.addpoly()
            sim.fdtd.set('name', poly_name)"""
        sim.fdtd.setnamed(poly_name, 'x', 0.0)
        sim.fdtd.setnamed(poly_name, 'y', 0.0)
        sim.fdtd.setnamed(poly_name, 'z', self.z)
        sim.fdtd.setnamed(poly_name, 'z span', self.depth)
        sim.fdtd.setnamed(poly_name, 'vertices', points)
        self.eps_in.set_script(sim, poly_name)

    def add_geo(self, sim, params, only_update):
        ''' Adds the geometry to a Lumerical simulation'''
        if params is None:
            points = self.points
        else:
            points = self.func(params)
        sim.fdtd.switchtolayout() # probably here, add litho params instead
        # use litho params instead
        self.add_poly_script(sim, points, only_update)
