from lumopt.geometries.geometry import Geometry
from lumopt.utilities.materials import Material
from lumopt.lumerical_methods.lumerical_scripts import set_spatial_interp, get_eps_from_sim

import lumapi
import numpy as np
import scipy as sp
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from lumopt.geometries.topology import TopologyOptimization2DParameters
from copy import deepcopy

eps0 = sp.constants.epsilon_0

class Topology2D_GL_Params(TopologyOptimization2DParameters):
    """ 
        A polygon object that supports lithography transformation and penalty terms for freeform optimization.
        
        New Parameters
        ----------
        :param lithography_model:      flag to use lithography model specified in the lithography class
        :param discreteness_threshold: percentage of discreteness to determine when to apply the lithography filter
    """

    def __init__(self, params, eps_min, eps_max, x, y, z, filter_R, eta, beta, eps=None, min_feature_size = 0, lithography_model = None, discreteness_threshold = 0.7):
        super().__init__(params, eps_min, eps_max, x, y, z, filter_R, eta, beta, eps, min_feature_size = min_feature_size)
        self._lithography_model = lithography_model
        self.discreteness_threshold = discreteness_threshold
        
    def update_geometry(self, params, sim):
            self.eps = self.get_eps_from_params(sim, params)
            temp_eps = deepcopy(self.eps)
            self.discreteness = self.calc_discreteness()
            if self._lithography_model and self.discreteness > self.discreteness_threshold:
                self.eps2 = self._lithography_model.apply_lithography_topology(self.eps)
                self.compare_eps_matrices(temp_eps, self.eps2)
                
    def compare_eps_matrices(self, temp_eps, eps):
        """
        Visually compare the temp_eps (before lithography transformation) and eps (after transformation)
        by displaying both images and a difference map overlay.
        
        Parameters:
        temp_eps (numpy.ndarray): The original matrix before lithography transformation.
        eps (numpy.ndarray): The transformed matrix after lithography is applied.
        """
        # Create a figure with three subplots: temp_eps, eps, and the difference map
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot temp_eps (before transformation)
        axes[0].imshow(temp_eps, cmap='gray')
        axes[0].set_title('Before Lithography (temp_eps)')
        axes[0].axis('off')  # Hide the axis

        # Plot eps (after transformation)
        axes[1].imshow(eps, cmap='gray')
        axes[1].set_title('After Lithography (eps)')
        axes[1].axis('off')  # Hide the axis

        # Compute the difference map (eps - temp_eps)
        difference = eps - temp_eps

        # Plot the difference map with a diverging colormap (e.g., 'RdBu')
        im = axes[2].imshow(difference, cmap='RdBu', vmin=-np.max(np.abs(difference)), vmax=np.max(np.abs(difference)))
        axes[2].set_title('Difference (eps - temp_eps)')
        axes[2].axis('off')  # Hide the axis

        # Add a color bar to indicate the scale of the differences
        fig.colorbar(im, ax=axes[2])

        # Display the plots
        plt.tight_layout()
        plt.show()

                    
    @property
    def lithography_model(self):
        return self._lithography_model

    @lithography_model.setter
    def lithography_model(self, model):
        self._lithography_model = model 
                
class TopologyOptimizationGL_2D(Topology2D_GL_Params):
    '''
    '''
    self_update = False

    def __init__(self, params, eps_min, eps_max, x, y, z=0, filter_R=200e-9, eta=0.5, beta=1, eps=None, min_feature_size = 0):
        super().__init__(params, eps_min, eps_max, x, y, z, filter_R, eta, beta, eps, min_feature_size = min_feature_size)


    @classmethod
    def from_file(cls, filename, z=0, filter_R=200e-9, eta=0.5, beta = None):
        data = np.load(filename)
        if beta is None:
            beta = data["beta"]
        return cls(data["params"], data["eps_min"], data["eps_max"], data["x"], data["y"], z = z, filter_R = filter_R, eta=eta, beta=beta, eps=data["eps"])

    def set_params_from_eps(self,eps):
        # Use the permittivity in z-direction. Does not really matter since this is just used for the initial guess and is (usually) heavily smoothed
        super().set_params_from_eps(eps[:,:,0,0,2])


    def calculate_gradients_on_cad(self, sim, forward_fields, adjoint_fields, wl_scaling_factor):
        lumapi.putMatrix(sim.fdtd.handle, "wl_scaling_factor", wl_scaling_factor)
        ## I like how theres zero fucking documentation for this very long script
        sim.fdtd.eval("V_cell = {};".format(self.dx*self.dy) +
                      "dF_dEps = pinch(sum(2.0 * V_cell * eps0 * {0}.E.E * {1}.E.E,5),3);".format(forward_fields, adjoint_fields) +
                      "num_wl_pts = length({0}.E.lambda);".format(forward_fields) +
                      
                      "for(wl_idx = [1:num_wl_pts]){" +
                      "    dF_dEps(:,:,wl_idx) = dF_dEps(:,:,wl_idx) * wl_scaling_factor(wl_idx);" +
                      "}" + 
                      "dF_dEps = real(dF_dEps);")

        rho = self.get_current_params_inshape()

        ## Expand symmetry (either along x- or y-direction)
        rho = self.unfold_symmetry_if_applicable(rho)

        sim.fdtd.putv("topo_rho", rho)
        sim.fdtd.eval(('params = struct;'
                       'params.eps_levels=[{0},{1}];'
                       'params.filter_radius = {2};'
                       'params.beta = {3};'
                       'params.eta = {4};'
                       'params.dx = {5};'
                       'params.dy = {6};'
                       'params.dz = 0.0;'
                       'dF_dp = topoparamstogradient(params,topo_rho,dF_dEps);').format(self.eps_min,self.eps_max,self.filter_R,self.beta,self.eta,self.dx,self.dy) )
        topo_grad = sim.fdtd.getv("dF_dp")

        ## Fold symmetry again (on the CAD this time)
        if self.symmetry_x:
            ## To be tested!
            sim.fdtd.eval(('dF_dp = dF_dp({0}:end,:,:);'
                           'dF_dp(2:end,:,:) = 2*dF_dp(:,2:end,:);').format(int((shape[0]+1)/2)))
        if self.symmetry_y:
            shape = rho.shape
            sim.fdtd.eval(('dF_dp = dF_dp(:,{0}:end,:);'
                           'dF_dp(:,2:end,:) = 2*dF_dp(:,2:end,:);').format(int((shape[1]+1)/2)))

        return "dF_dp"


    def calculate_gradients(self, gradient_fields, sim):

        rho = self.get_current_params_inshape()

        # If we have frequency data (3rd dim), we need to adjust the dimensions of epsilon for broadcasting to work
        E_forward_dot_E_adjoint = np.atleast_3d(np.real(np.squeeze(np.sum(gradient_fields.get_field_product_E_forward_adjoint(),axis=-1))))

        dF_dEps = 2*self.dx*self.dy*eps0*E_forward_dot_E_adjoint
        
        sim.fdtd.putv("topo_rho", rho)
        sim.fdtd.putv("dF_dEps", dF_dEps)
        sim.fdtd.eval(('params = struct;'
                       'params.eps_levels=[{0},{1}];'
                       'params.filter_radius = {2};'
                       'params.beta = {3};'
                       'params.eta = {4};'
                       'params.dx = {5};'
                       'params.dy = {6};'
                       'params.dz = 0.0;'
                       'topo_grad = topoparamstogradient(params,topo_rho,dF_dEps);').format(self.eps_min,self.eps_max,self.filter_R,self.beta,self.eta,self.dx,self.dy) )
        topo_grad = sim.fdtd.getv("topo_grad")

        return topo_grad.reshape(-1, topo_grad.shape[-1])


    def add_geo(self, sim, params=None, only_update = False):

        fdtd=sim.fdtd

        eps = self.eps if params is None else self.get_eps_from_params(sim, params.reshape(-1))

        fdtd.putv('x_geo',self.x)
        fdtd.putv('y_geo',self.y)
        fdtd.putv('z_geo',np.array([self.z-self.depth/2,self.z+self.depth/2]))

        if not only_update:
            set_spatial_interp(sim.fdtd,'opt_fields','specified position') 
            set_spatial_interp(sim.fdtd,'opt_fields_index','specified position') 

            script=('select("opt_fields");'
                    'set("x min",{});'
                    'set("x max",{});'
                    'set("y min",{});'
                    'set("y max",{});').format(np.amin(self.x),np.amax(self.x),np.amin(self.y),np.amax(self.y))
            fdtd.eval(script)

            script=('select("opt_fields_index");'
                    'set("x min",{});'
                    'set("x max",{});'
                    'set("y min",{});'
                    'set("y max",{});').format(np.amin(self.x),np.amax(self.x),np.amin(self.y),np.amax(self.y))
            fdtd.eval(script)

            script=('addimport;'
                    'set("detail",1);')
            fdtd.eval(script)

            mesh_script=('addmesh;'
                        'set("x min",{});'
                        'set("x max",{});'
                        'set("y min",{});'
                        'set("y max",{});'
                        'set("dx",{});'
                        'set("dy",{});').format(np.amin(self.x),np.amax(self.x),np.amin(self.y),np.amax(self.y),self.dx,self.dy)
            fdtd.eval(mesh_script)

        if eps is not None:
            fdtd.putv('eps_geo',eps)

            ## We delete and re-add the import to avoid a warning
            script=('select("import");'
                    'delete;'
                    'addimport;'
                    'temp=zeros(length(x_geo),length(y_geo),2);'
                    'temp(:,:,1)=eps_geo;'
                    'temp(:,:,2)=eps_geo;'
                    'importnk2(sqrt(temp),x_geo,y_geo,z_geo);')
            fdtd.eval(script)
            

