import os
import inspect
import numpy as np
from lumopt.optimization import Optimization
from lumopt.geometries.geometry import Geometry
from lumopt.geometries.polygon import Polygon
from lumopt_gl.geometries.topology_gl import Topology2D_GL_Params
from lumopt_gl.geometries.geometry_gl import Geometry_GL
from lumopt.utilities.base_script import BaseScript
from lumopt.utilities.wavelengths import Wavelengths
from lumopt.utilities.simulation import Simulation
from lumopt.lumerical_methods.lumerical_scripts import get_fields, get_fields_on_cad, get_lambda_from_cad
from config import Config

class OptimizationGL(Optimization):
    """
    Extension of the Optimization class to support GPU, lithography, and remove some redundancies.

    For more information, see the documentation in `lumopt.optimization`.

    Args:
        GPU (bool): Flag to use GPU acceleration (for 3D simulations only).
        use_deps_zip (bool): Flag to add all geometries to one GDS file and use the lithography model only once during DEP calculation.
    """
    
    def __init__(self, base_script, wavelengths, fom, geometry, optimizer, use_var_fdtd = False, hide_fdtd_cad = False, use_deps = True, plot_history = True, store_all_simulations = True, save_global_index = False, label=None, source_name = 'source', fields_on_cad_only = False, GPU = True, use_deps_zip = True):
        super().__init__(base_script, wavelengths, fom, geometry, optimizer, use_var_fdtd, hide_fdtd_cad, use_deps, plot_history, store_all_simulations, save_global_index, label, source_name, fields_on_cad_only)
        # Must rerun this section to prevent files being stored in the wrong directory
        frame = inspect.stack()[1]
        self.calling_file_name = os.path.abspath(frame[0].f_code.co_filename)
        self.base_file_path = os.path.dirname(self.calling_file_name)
        self.GPU = GPU
        self.use_deps_zip = use_deps_zip
        self.use_var_fdtd = use_var_fdtd
            
    def initialize(self, working_dir=None):
        """Extension of initialize to support GPU."""
        if working_dir == None:
            working_dir=f'{Config.RESULTS_PATH}/opts'
        super().initialize(working_dir)
        if not self.use_var_fdtd:
            dimension = self.sim.fdtd.getnamed("FDTD", "dimension")
            if self.GPU and dimension == '3D':
                self.sim.fdtd.setresource("FDTD","GPU", True)
                self.sim.fdtd.setnamed("FDTD", "express mode", True)
            else:
                self.sim.fdtd.setnamed("FDTD", "express mode", False)
                self.sim.fdtd.setresource("FDTD","GPU", False)
                
    def make_adjoint_sim(self, params, iter,co_optimizations = None, one_forward = False):
        """Removes the duplicate polygon generation for the adjoint simulation."""
        print(self.geometry.get_current_params())
        assert np.allclose(params, self.geometry.get_current_params())
        if one_forward:
            adjoint_name = 'adjoint_{0}_{1}'.format(co_optimizations.index(self),iter)
            self.sim = co_optimizations[0].sim
        else:
            adjoint_name = 'adjoint_{}'.format(iter)
        self.sim.fdtd.switchtolayout()
        self.sim.fdtd.setnamed(self.source_name, 'enabled', False)
        self.fom.make_adjoint_sim(self.sim)
        if  co_optimizations is not None and len(co_optimizations) > 1 and one_forward:
            for co_opt in co_optimizations:
                if co_opt != self:
                    self.sim.fdtd.setnamed(co_opt.fom.adjoint_source_name, 'enabled', False)
        return self.sim.save(adjoint_name)
    
    def callable_jac(self, params):
        """Extends function to multiply gradients with lithography jacobian for filter and ML models."""
        self.sim.fdtd.clearjobs()
        iter = self.optimizer.iteration if self.store_all_simulations else 0
        no_forward_fields = not hasattr(self,'forward_fields')
        params_changed = not np.allclose(params, self.geometry.get_current_params())
        redo_forward_sim = no_forward_fields or params_changed
        do_adjoint_sim = redo_forward_sim or not self.optimizer.concurrent_adjoint_solves() or self.forward_fields_iter != iter 
        if redo_forward_sim:
            print('Making forward solve')
            forward_job_name = self.make_forward_sim(params, iter)
            self.sim.fdtd.addjob(forward_job_name)
        if do_adjoint_sim:
            print('Making adjoint solve')
            adjoint_job_name = self.make_adjoint_sim(params, iter)
            self.sim.fdtd.addjob(adjoint_job_name)
        if len(self.sim.fdtd.listjobs()) > 0:
            print('Running solves')
            self.sim.fdtd.runjobs()
        if redo_forward_sim:
            print('Processing forward solve')
            fom = self.process_forward_sim(iter)
        print('Processing adjoint solve')
        self.process_adjoint_sim(iter)
        print('Calculating gradients')
        grad = self.calculate_gradients()
        
        # multiply with jacobian if using lithography
        try:
            if self.geometry.lithography_model.model in ["GaussianDUV", "GaussianEBL", "ML_Model"] and self.geometry.discreteness > self.geometry.discreteness_threshold:
                grad = self.geometry.lithography_model.apply_jacobian(grad)
        except AttributeError:
            pass
        self.last_grad = grad

        if hasattr(self.geometry,'calc_penalty_term'):
            print('Calculating Penalty Terms')
            penalty_grad = self.geometry.calc_penalty_gradient(self.sim, params)
            grad += penalty_grad
        return grad
    
    def run_forward_simulation(self, params = None, eps = None):
        # assumes the opt object is already initialized.
        self.sim.fdtd.switchtolayout()
        self.sim.fdtd.save("temp_simulation_file.lms")  # Save the file to a temporary file if needed
        
        if params is None:
            if isinstance(self.geometry, Polygon):
                params = self.geometry.current_params
            if isinstance(self.geometry, Topology2D_GL_Params):
                return self.run_forward_simulation_topology(eps)
            else:
                params = self.geometry.params
        
        self.geometry.update_geometry(params, self.sim)
        self.geometry.add_geo(self.sim, params = None, only_update = True)
        Optimization.deactivate_all_sources(self.sim)
        self.sim.fdtd.setnamed(self.source_name, 'enabled', True)
        self.sim.fdtd.run()
        
        self.forward_fields = get_fields(self.sim.fdtd,
                                    monitor_name = 'opt_fields',
                                    field_result_name = 'forward_fields',
                                    get_eps = True,
                                    get_D = not self.use_deps,
                                    get_H = False,
                                    nointerpolation = not self.geometry.use_interpolation(),
                                    unfold_symmetry = self.unfold_symmetry)
        fom =  self.fom.get_fom(self.sim)
        self.sim.remove_data_and_save()
        return fom
    
    def run_forward_simulation_topology(self, eps):
        self.sim.fdtd.save("temp_simulation_file.lms")  # Save the file to a temporary file if needed
        self.geometry.eps = eps
        self.geometry.add_geo(self.sim, params = None, only_update = True)
        Optimization.deactivate_all_sources(self.sim)
        self.sim.fdtd.setnamed(self.source_name, 'enabled', True)
        self.sim.fdtd.run()
        self.forward_fields = get_fields(self.sim.fdtd,
                                    monitor_name = 'opt_fields',
                                    field_result_name = 'forward_fields',
                                    get_eps = True,
                                    get_D = not self.use_deps,
                                    get_H = False,
                                    nointerpolation = not self.geometry.use_interpolation(),
                                    unfold_symmetry = self.unfold_symmetry)
        
        fom =  self.fom.get_fom(self.sim)
        self.sim.remove_data_and_save()
        return fom
    
    def plot_gradient(self, fig, ax1, ax2):
        try:
            self.gradient_fields.plot(fig, ax1, ax2)
        except:
            pass
        
    def plotting_function(self, params):
        ## Add the last FOM evaluation to the list of FOMs that we wish to plot. This removes entries caused by linesearches etc.
        self.fom_hist.append(self.full_fom_hist[-1])

        ## In a multi-FOM optimization, only the first optimization has a plotter
        if self.plotter is not None:

            self.params_hist.append(params)
            
            try:
                self.grad_hist.append(self.last_grad / self.optimizer.scaling_factor)
            except:
                pass

            self.plotter.clear()
            self.plotter.update_fom(self)
            self.plotter.update_gradient(self)
            self.plotter.update_geometry(self)
            self.plotter.draw_and_save()

            self.save_index_to_vtk(self.optimizer.iteration)

            if hasattr(self.geometry,'to_file'):
                self.geometry.to_file(os.path.join(self.workingDir,'parameters_{}.npz').format(self.optimizer.iteration))

            with open(os.path.join(self.workingDir,'convergence_report.txt'),'a') as f:
                f.write('{}, {}'.format(self.optimizer.iteration,self.fom_hist[-1]))

                if hasattr(self.geometry,'write_status'):
                    self.geometry.write_status(f) 

                if len(self.params_hist[-1])<250:
                    f.write(', {}'.format(np.array2string(self.params_hist[-1], separator=', ', max_line_width=10000)))

                try:
                    if len(self.grad_hist[-1])<250:
                        f.write(', {}'.format(np.array2string(self.grad_hist[-1], separator=', ', max_line_width=10000)))
                except:
                    pass

                f.write('\n')
    
    """def _initalize_lithography(self):
        ## REVIEW: Probably unnecessary with the setter function.
        if self.geometry.lithography_model is not None:
            self.geometry.litho_success = True # start off with True
            if self.geometry.lithography_model.model == 'DUV' and self.use_deps_zip:
                self.geometry.use_deps_zip = True"""