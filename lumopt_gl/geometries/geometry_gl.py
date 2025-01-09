import sys
import numpy as np
import lumapi
from lumopt.geometries.geometry import Geometry
import time as time
from lumopt_gl.geometries.polygon_gl import Polygon_GL
from lumopt.geometries.polygon import Polygon

class Geometry_GL(Geometry):
    """
    Represents and manipulates geometrical shapes for simulations.

    Extends the existing Geometry class to smoothly handle multi-polygon geometries, especially with shared parameters.
    Supports GDS-based lithography models where the polygons use permittivity perturbation (use_deps = True).
    If your workflow is differential (use_deps = False), lithography should be applied to those topology_gl/polygon_gl.

    Args:
        geometries (list): Geometries or polygons to be passed to this object.
            NOTE: Geometry_GL flattens all nested geometries to polygons and uses dicts for mapping.
        lithography_model (bool, optional): Flag to use lithography model during optimization. Default is None.
            Forced to None if use_deps = False. Apply lithography to polygon directly instead.
        operation (str): Specifies how the geometries are created by adding, multiplying, or merging polygons:
            - 'mul': All polygons share the same optimization parameters.
            - 'add': All polygons have unique parameters.
            - 'merge': Polygons contain some shared and some unique parameters.
    """
    def __init__(self,geometries,operation, lithography_model = None):
        self.operation=operation
        self.geometries = []
        self.params = []
        self.mapping = {}
        self.bounds = []
        self._flatten_geometries(geometries)
        if self.operation=='mul':
            self.bounds=geometries[0].bounds
        self.dx=max([geo.dx for geo in self.geometries])
        
        self._lithography_model = lithography_model
        self.use_deps_zip = False # Flag to zip perturbed geometries before running lithography model with deps method.
        self.litho_success = False

    def add_geo(self, sim, params, only_update, d_index = None):
        """Extension of the add geo operation to support lithography."""
        current_index = 0
        if self._lithography_model and self.litho_success:
            self._disable_polygons(sim) # usually not necessary. But to be safe.
            polygons = [geo for geo in self.geometries]
            self.litho_success = self._lithography_model.apply_lithography_GDS(sim, polygons)
            
        # add ideal geometry if the lithography model has failed or lithography model is not being used.
        if not self.litho_success:
            if d_index:
                affected_polygons = [polygon for polygon, indices in self.mapping.items() if d_index in indices or (d_index-1) in indices]
                for polygon in affected_polygons:
                    polygon.add_geo(sim, None, only_update)
            else:
                for polygon in self.geometries:
                    polygon.add_geo(sim, None, only_update)
        
    def update_geometry(self, params, sim = None):
        """Extension of the update geometry operation to support multiple polygons and lists of polygons"""
        
        self.params = params
        for geometry in self.geometries:
            specific_params = [params[i] for i in self.mapping[geometry]]  # Extract specific parameters for this geometry
            geometry.update_geometry(specific_params, sim)
        
    def get_current_params(self):
        # ah.. so.. current params has been updated. 
        # but not the params of this geo..
        """all_params = []
        for geometry in self.geometries:
            all_params.extend(geometry.get_current_params())"""
        return np.array(self.params)
    
    def d_eps_on_cad_serial(self, sim):
        """Extension of the deps method to support lithography and better multipolygon handling"""

        # Deps_zip is the preferred model for GDS lithography. It zips perturbed geometries to one GDS.
        if self._lithography_model and self.use_deps_zip:
            deps_geometries = [] # restart each time
            
        sim.fdtd.redrawoff()

        Geometry.get_eps_from_index_monitor(sim.fdtd, 'original_eps_data')
        current_params = self.get_current_params()
        sim.fdtd.eval("d_epses = cell({});".format(current_params.size))
        cur_dx = self.dx

        lumapi.putDouble(sim.fdtd.handle, "dx", cur_dx)
        print('Getting d eps: dx = ' + str(cur_dx))

        for i,param in enumerate(current_params):
            d_params = current_params.copy()
            d_params[i] = param + cur_dx
            self.update_geometry(d_params, sim) # necessary for litho
            
            if not self.use_deps_zip or not self.litho_success:
                # if not using deps_zip, or if litho failed for this geometry, we perturb and add the geometries one at a time
                self.add_geo(sim, d_params, only_update = True, d_index=i) # we only need to reconstruct geos which actually changed
                Geometry.get_eps_from_index_monitor(sim.fdtd, 'current_eps_data')
                sim.fdtd.eval("d_epses{"+str(i+1)+"} = (current_eps_data - original_eps_data) / dx;")
                sys.stdout.write('.'), sys.stdout.flush()
            else:
                # else, add the updated polygons to the list of geometries
                polygon_points = [poly.points.copy() for poly in self.geometries]
                deps_geometries.append(polygon_points)
        
        # now process the deps_geometries if necessary
        if self.use_deps_zip and self.litho_success:
            self._deps_from_zipped_gds(sim,deps_geometries,current_params.size)
            
        sim.fdtd.eval("clear(original_eps_data, current_eps_data, dx);")
        time.sleep(1.0) # give enough time to clear data. Otherwise varFDTD suffers memory overflow.
        sim.fdtd.redrawon()
        if self._lithography_model:
            self.litho_success = True # once complete, try lithography again regardless of failure, as the geometry will be new

    def _flatten_geometries(self, geometries):
        """ Flatten all geometries to handle nested structures and multiple polygons. """
        # i think all of this is unnecessary lol
        # Process polygons after flattening based on the operation
        if self.operation == 'mul':
            self._handle_mul_operation(geometries)
        elif self.operation == 'add':
            self._handle_add_operation(geometries)

    def _handle_mul_operation(self, geometries):
        """ All polygons use the same optimization parameters, based on the first polygon's parameters. """
        # Use the parameters of the first geometry as the standard
        standard_params = geometries[0].current_params

        # Check if all geometries have the same number of parameters
        for poly in geometries:
            if len(poly.current_params) != len(standard_params):
                raise ValueError("All geometries must have the same number of parameters for 'mul' operation")

        # Assign parameters from the first geometry to all others
        self.params = list(standard_params)
        for poly in geometries:
            self.geometries.append(poly)
            self.mapping[poly] = list(range(len(standard_params)))  # Directly use index mapping

    def _handle_add_operation(self, geometries):
        """ Each polygon has unique parameters. """
        current_param_index = len(self.params)  # Start from the current end of the parameter list
        for geometry in geometries:
            if isinstance(geometry, Polygon):
                num_params = len(geometry.current_params)
                start_idx = current_param_index  
                self.params.extend(geometry.current_params)
                self.geometries.append(geometry)
                self.mapping[geometry] = list(range(start_idx, start_idx + num_params))
                current_param_index += num_params
                self.bounds.extend(geometry.bounds)
            if isinstance(geometry, Geometry_GL):
                for poly, indices in geometry.mapping.items():
                    adjusted_indices = [current_param_index + idx for idx in indices]
                    self.mapping[poly] = adjusted_indices
                    self.geometries.append(poly)
                self.bounds.extend(geometry.bounds)
                self.params.extend(geometry.params)
                current_param_index += len(geometry.params)
                
    def _disable_polygons(self, sim):
        """disable polygons if necessary"""
        try:
            for polygon in self.geometries:
                sim.fdtd.setnamed(f'polygon_{polygon.hash}', 'enabled', 0)
        except:
            pass
    
    def _deps_from_zipped_gds(self, sim, deps_geometries, size):
        """Method to extract perturbation information from a list of perturbed geometries"""
        litho_gds = self._lithography_model.deps_zip_to_gds(deps_geometries) # add all deps_geometries to a GDS
        result_gds_list = self._lithography_model.extract_deps_gds(litho_gds,size) # generate a list of GDS files
        for i,gds_dir in enumerate(result_gds_list):
            # extract gds one by one and perform deps calculation
            import_success = self._lithography_model.import_gds(sim, gds_dir)
            if import_success:
                Geometry.get_eps_from_index_monitor(sim.fdtd, 'current_eps_data')
                time.sleep(.4)
                sim.fdtd.eval("d_epses{"+str(i+1)+"} = (current_eps_data - original_eps_data) / dx;")
            else:
                break # deps_zip import cancelled
                                 
    def __add__(self,other):
        """Extension of the add operation to support multiple polygons and lists of polygons"""
        if isinstance(other, Geometry):
            geometries = self.geometries + [other]
        elif isinstance(other, list): 
            geometries = self.geometries + other
        geometries = self._flatten_geometries(geometries)
        return Geometry(geometries, 'add')
    
    @property
    def lithography_model(self):
        return self._lithography_model

    @lithography_model.setter
    def lithography_model(self, model):
        self._lithography_model = model
        self.litho_success = bool(model)
        if self._lithography_model is not None and self._lithography_model.model == 'DUV':
            self.use_deps_zip = True
        else:
            self.use_deps_zip = False
