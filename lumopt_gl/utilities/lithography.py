from config import Config
import paramiko
import scp
import time
import pya
import gdspy
import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn.functional as F
    from torch.autograd.functional import jacobian
    import torchvision.transforms as transforms
    pytorch_loaded = True
except ImportError:
    pytorch_loaded = False

try:
    import prefab as pf
except ImportError:
    pf = None

try:
    import tensorflow as tf
    tensorflow_loaded = True
except ImportError:
    tensorflow_loaded = False
    
from lumopt.lumerical_methods.lumerical_scripts import set_spatial_interp, get_eps_from_sim

class LithographyOptimizer:
    def __init__(self, model, **kwargs):
        """
        Args:
            model (str): The lithography model you wish to use. Supported models are listed below.
                For Parameterized Optimization:
                    - 'DUV': Utilizes Lin's model for the DUV process.
                    - 'NanoSOI': Utilizes PreFab ML-based model for ANT Foundry NanoSOI process.
                    - 'SiN': Utilizes PreFab ML-based model for ANT SiN process.
                For Topology Optimization:
                    - 'GaussianDUV': Utilizes a tuned Gaussian Filter for the DUV process.
                    - 'GaussianEBL': Utilizes a tuned Gaussian Filter for the EBL process.
                    - 'ML_Model': Utilizes a trained PyTorch model. Requires the trained model
                    - 'Prefab_Diff': NOTE: Needs testing. 

        Kwargs:
            ml_model (Any): The trained PyTorch or TensorFlow model for 'ML_Model'. Required for 'ML_Model'.
            gds_layer (int): The GDS layer number. Default is 1.
            gds_litho_data_type (int): The GDS lithography data type. Default is 69.
            gds_data_type (int): The GDS data type. Default is 0.
            target_size (tuple): the original size of the topology optimization device. Required for old prefab...
            rasterize (bool): Flag to use experimental rasterization and edge detection for boundary perturbation.
            prefab_model (str): If this argument is passed the optimizer will use this prefab model.
            slice_length (int): old prefab model slicing length. Default is 128
            step_length (str): old prefab model step length. Default is 64
        """
        self._validate_model(model)
        self.model = model
        self.ssh = None
        self.device = None
        self.eps_in = None
        self.ml_model = kwargs.get('ml_model', None)
        self.gds_layer = kwargs.get('gds_layer', 1)
        self.gds_data_type = kwargs.get('gds_data_type', 0)
        self.target_size = kwargs.get('target_size', (3000,1800))
        self.gds_litho_data_type = kwargs.get('gds_litho_data_type', 69)
        self.prefab_model = kwargs.get('prefab_model', None)
        self.slice_length = kwargs.get('slice_length', 128)
        self.step_length = kwargs.get('step_length', 64)
        self.rasterize = kwargs.get('rasterize', False)
        self._configure_model() 
    
    def set_model(self, model, geometry):
        """ Set a new model for the lithography process and reinitialize"""
        self._validate_model(model)
        self.model = model
        self._configure_model()
        geometry.litho_success = True
            
    def apply_lithography_GDS(self, sim, polygons):
        """
        Applies lithography based on the specified model for GDS based models. 

        Parameters:
            sim (Simulation): The simulation instance where the GDS geometry is imported.
            polygons (list): A list of polygons representing the ideal geometry.

        Returns:
            bool: True if the GDS geometry was successfully imported into the simulation, False otherwise.
        """
        gds_path = self.write_to_gds(polygons)
        if self.eps_in is None:
            self._extract_geo_params(polygons)
        if self.model == 'DUV':
            result_path = self.run_DUV_model(gds_path)
        elif self.model in ['SiN', 'NanoSOI']:
            result_path = self.run_PreFab_model(gds_path)
            self._prefab_centering(polygons,result_path)
        if result_path:
            return self.import_gds(sim, result_path)
        else:
            return False
        
    def run_DUV_model(self, gds_path):
        """
        Executes DUV lithography on geometries in the lithography layer of the GDS file.
        Resulting lithography shape is assumed to be placed the default GDS datatype (0)
        Model is accessed through SSH to UBC servers
        This method can be repurposed for use with other server based lithography models.
        
        Parameters:
            gds_path (str): path to GDS file containing ideal geometry.
            
        Returns:
            result_path (str): path to GDS file containing lithography output geometry.
        """
        if not self.ssh.get_transport().is_active():
            print("SSH connection not active. Attempting to connect...")
            self._connect_ssh()
            
        result_path = f'{Config.RESULTS_PATH}/result.gds'
        self.scp_client.put(gds_path,f'{Config.MODEL_PATH}/input.gds')
        stdin, stdout, stderr = self.ssh.exec_command(Config.SSH_COMMAND)
        stderr_output = stderr.read().decode()
        
        if stderr_output:
            print(f'Error: {stderr_output}. Server Lithography model failed to run.')
            return False
        
        self.scp_client.get(f'{Config.GDS_PATH}', result_path)
        return result_path
    
    def run_PreFab_model(self, gds_path):
        """
        Uses PreFab predictor on geometries in the lithography layer of the GDS file.
        Resulting lithography shape is assumed to be placed the default GDS datatype (0)
        
        Parameters:
            gds_path (str): path to GDS file containing ideal geometry.
            
        Returns:
            result_path (str): path to GDS file containing lithography output geometry.
        """
        # REVIEW, support image input
        device = pf.read.from_gds(gds_path, "Top", gds_layer=(self.gds_layer, self.gds_litho_data_type))
        device.plot()
        prediction = device.predict(model=pf.models[self.prefab_model])
        result_path = f'{Config.RESULTS_PATH}/result.gds'
        device.to_gds(gds_path=result_path,
                      cell_name='Top', 
                      gds_layer=(self.gds_layer, self.gds_data_type), 
                      contour_approx_mode=2)
        return result_path
    
    def apply_lithography_boundary(self, sim, params):
        """
        Applies a specified filter to the boundary parameters and returns the filtered parameters.
        
        Parameters:
            parameters (list): Y-values of the boundary.
        
        Returns:
            filtered_params: The filtered y-values
            
        """
        # vector or image model?
        if self.model in 'VectorNet':
            params_tensor = torch.tensor(params, dtype=torch.double, requires_grad=True)
            self.output_tensor = self.ml_model(params_tensor)
        else:
            # if image, it must first be rasterized
            eps,x,y,z,l = get_eps_from_sim(sim) # get EPS from the GDS
            eps_tensor = torch.tensor(eps, dtype=torch.double, requires_grad=True).unsqueeze(0).unsqueeze(0)
            
            #then filtered
            if self.model in 'GaussianDUV':
                filter = transforms.GaussianBlur(kernel_size=15, sigma=(5, 5))
                blurred_tensor = filter(eps_tensor)
                self.output_tensor = torch.sigmoid(blurred_tensor)
                
            elif self.model in 'GaussianEBL':
                filter = transforms.GaussianBlur(kernel_size=5, sigma=(1.5, 1.5))
                blurred_tensor = filter(eps_tensor)
                self.output_tensor = torch.sigmoid(blurred_tensor)
                
            elif self.model in 'ML_Model':
                # define the forward transformation of the ML model
                blurred_tensor = self.ml_model(eps_tensor)
                self.output_tensor = torch.sigmoid(blurred_tensor)
                
            elif self.model in 'Prefab_Diff':
                # for old prefab (in the new one the model cannot be backpropagated) we're gonna split it to 128x128 slices where each pixel represents a nm in the simulation
                eps_max = np.max(eps)
                prediction = self._prefab_forward_run_img(eps, eps_max)

                # now we need to convert prediction back to an eps
                self.blurred_tensor = self.convert_prefab_predict_to_eps(prediction, eps_max)
                self.output_tensor = torch.sigmoid(blurred_tensor)
                return self.output_tensor
            
            return self.output_tensor.squeeze(0).squeeze(0).detach().numpy()
                 
    def apply_lithography_topology(self, eps):
        """
        Applies a specified filter to the input tensor and calculates the Jacobian matrix if not already initialized.
        
        Parameters:
            eps (numpy.ndarray): The input 2D matrix of topology optimization permittivity data at every x,y value
        
        Returns:
            numpy.ndarray: The filtered output matrix.
        """
        
        eps_tensor = torch.tensor(eps, dtype=torch.double, requires_grad=True).unsqueeze(0).unsqueeze(0)

        # Define the filter
        if self.model in 'GaussianDUV':
            filter = transforms.GaussianBlur(kernel_size=15, sigma=(5, 5))
            self.output_tensor = filter(eps_tensor) # Apply the forward transformation
            
        elif self.model in 'GaussianEBL':
            filter = transforms.GaussianBlur(kernel_size=5, sigma=(1.5, 1.5))
            self.output_tensor = filter(eps_tensor) # Apply the forward transformation
            
        elif self.model in 'ML_Model':
            # define the forward transformation of the ML model
            filter = self.ml_model(eps_tensor)
            self.output_tensor = filter(eps_tensor) # Apply the forward transformation
            
        elif self.model in 'Prefab_Diff':
            # prediction = self._prefab_forward_run_img(eps, eps_max)

            # NOT TESTED
            self.output_tensor = pf.predict.predict_array_with_grad(eps, self.prefab_model)

            return self.output_tensor
        
        return self.output_tensor.squeeze(0).squeeze(0).detach().numpy()
    
    def apply_jacobian(self, grad):
        """
        Applies the Jacobian (dY/dX) to the gradient tensor (dF/dY) and returns the result (dF/dX).
        
        Parameters:
        grad (numpy.ndarray): a flattened array of gradient data at every x,y parameter for topology optimization
        
        Returns:
        numpy.ndarray: The resulting flattened array after applying the Jacobian to the input gradient.
        """
        if self.model in 'Prefab_Diff':
            # NOTE: NOT TESTED
            vjp_function = predict_array_with_grad_vjp(self.prediction, self.device_array)

            # Compute and return the result using the VJP function
            return vjp_function(grad)
        grad_tensor = torch.tensor(grad, dtype=torch.double).view_as(self.output_tensor)
        self.output_tensor.backward(gradient=grad_tensor)
        dF_dX_flattened = self.eps_tensor.grad.flatten().detach().numpy()
        return dF_dX_flattened
    
    def apply_jacobian_prefab(self, grad):
        ## DEPRECATED. Use JVP from prefab instead.
        pass

    def compute_jvp_prefab(self, grad):
        ## DEPRECATED. Use JVP from prefab instead.
        pass
        
    def deps_zip_to_gds(self, deps_polygons):
        """
        Writes a list of list of polygons generated for deps method to a GDSII file.
        Default method for DUV lithography model. Not used with PreFab GDS.

        Parameters:
            deps_polygons (list): A list of polygon lists

        Returns:
            str: The local path to the saved GDSII file.
        """
        layout = pya.Layout()
        top_cell = layout.create_cell("Top")
        
        self.height = self.extract_height(deps_polygons[0]) # required for translation
        
        for index, polygons in enumerate(deps_polygons):
            translation_y = index * 4 * self.height
            
            for points in polygons:
                klayout_points = [pya.Point(x * 1e9, y * 1e9) for x, y in points]
                points = pya.Polygon(klayout_points)
                
                # Translate the polygons vertically to place them for later extraction
                trans = pya.Trans(pya.Trans.R0, 0, translation_y)
                translated_polygon = points.transformed(trans)
                
                top_cell.shapes(layout.layer(self.gds_layer, self.gds_litho_data_type)).insert(translated_polygon)
        
        gds_path = f"{Config.RESULTS_PATH}/zip_gds.gds"
        layout.write(gds_path)
        
        litho_gds = self.run_DUV_model(gds_path) # run the DUV model on the list of perturbed geometries
        return litho_gds
    
    def extract_deps_gds(self, zip_gds, size):
        # Review: pretty messy. Rewrite...
        """
        Extension of the deps method to support lithography and better multipolygon handling
        Deps_zip method is the preferred method of applying lithography and is true unless explicitly set to false
        
        Parameters: 
            litho_gds: gds that contains all perturbed geometries.

        Returns: 
            result_gds_list (list): A list of gds files containing individual perturbed geometries
        """
        result_gds_list = []
        
        # Load the GDS file
        layout = pya.Layout()
        layout.read(zip_gds)
        top_cell = layout.top_cell()
        layer_index = layout.find_layer(self.gds_layer, self.gds_data_type)
        
        # Create a new layout for each translated polygon
        for i in range(size):
            # Define the y-coordinate range for selection
            y_start = i * self.height *4 - self.height * 2 # convert micrometers to nanometers
            y_end = y_start + self.height * 4

            # Select polygons in the specified range
            polygons = self.select_polygons_in_y_range(top_cell, layer_index, y_start, y_end)
            layout2 = pya.Layout()
            cell = layout2.create_cell("Top")
            for j, poly2 in enumerate(polygons):
                moved_polygon = poly2.transformed(pya.Trans(0, -1*self.height * 4*i))
                # Add the polygon to the cell
                cell.shapes(layout2.layer(self.gds_layer, self.gds_data_type)).insert(moved_polygon)
            file_name = rf"{Config.RESULTS_PATH}\deps_output_{i}.gds" # REVIEW
            layout2.write(file_name)
            result_gds_list.append(file_name)
        return result_gds_list
        
    def write_to_gds(self, geometry):
        """
        Writes geometry to a GDS file.

        Parameters:
            geometry (list): A list of polygons
        
        Returns:
            gds_path (str): path of the resulting GDS file.
        """
        layout = pya.Layout()
        top_cell = layout.create_cell("Top")
        for polygon in geometry:
            klayout_points = [pya.Point(x * 1e9, y * 1e9) for x, y in polygon.points]
            polygon = pya.Polygon(klayout_points)
            top_cell.shapes(layout.layer(self.gds_layer, self.gds_litho_data_type)).insert(polygon)

        gds_path = f'{Config.RESULTS_PATH}/input.gds'
        layout.write(gds_path)
        return gds_path
    
    def import_gds(self, sim, gds_path):
        """Attempts to import geometry to FDTD instance. Returns success status."""
        sim.fdtd.eval(f'select("GDS_LAYER_{self.gds_layer}:{self.gds_data_type}"); delete;')
        try:
            sim.fdtd.eval(f"gdsimport('{gds_path}','Top', '{self.gds_layer}:{self.gds_data_type}');")
            sim.fdtd.eval(f"set('z', {self.z}); set('z span', {self.depth});")
            self.eps_in.set_script(sim, f'GDS_LAYER_1:0')
            time.sleep(0.5) # necessary to add this delay before processing fields and perturbations. 
            return True
        except:
            sim.fdtd.eval(f'select("GDS_LAYER_{self.gds_layer}:{self.gds_data_type}"); delete;')
            print("Lithography Model GDS incompatible. Proceeding without lithography.")
            return False

    def _prefab_forward_run_img(self, eps):
        # deprecated. Use prefab JVP instead.
        pass
    
    def _validate_model(self, model):
        """ Ensures the model is among the supported models."""
        supported_models = ['DUV', 'NanoSOI', 'SiN', 'GaussianDUV', 'GaussianEBL']
        if model not in supported_models:
            raise ValueError(f"Unsupported model '{model}'. Supported models are: {', '.join(supported_models)}")
    
    def _configure_model(self):
        """Model dependent initialization."""
        if self.model == 'DUV':
            self._setup_ssh()
            
        if self.model in ['NanoSOI', 'SiN']:
            if pf is None:
                raise ImportError("The required PreFab module is missing. Please ensure it is installed.")
            if self.model == 'NanoSOI' and self.prefab_model == None:
                self.prefab_model = "ANT_NanoSOI_ANF1_d9"
            elif self.model == 'SiN' and self.prefab_model == None:
                self.prefab_model = "ANT_SiN_ANF1-d1"
        elif self.model in ['GaussianDUV', 'GaussianEBL']:
            if not pytorch_loaded:
                raise ImportError("PyTorch and torchvision are required for Gaussian models. Please install them to proceed.")
            else:
                self.jacobian = None
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif self.model == 'ML_Model':
            if self.ml_model is None:
                raise ValueError("Custom model type selected, but no ML model instance provided.")
            self._configure_custom_model()
        elif self.model == 'Prefab_Diff':
            if self.ml_model is None:
                raise ValueError("Custom model type selected, but no ML model instance provided.")
            self._configure_custom_model()
            
    def _configure_custom_model(self):
        """Checks the type of ML model passed and configures as needed."""
        if isinstance(self.ml_model, torch.nn.Module):
            # Check if PyTorch is properly installed and configure model for evaluation
            if not pytorch_loaded:
                raise ImportError("PyTorch is required for the provided ML model. Please install it to proceed.")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.ml_model.to(self.device)
            self.ml_model.eval()
        
        elif 'tensorflow' in str(type(self.ml_model)):
            # Check if tensorflow is properly installed and configure model for evaluation
            if not tensorflow_loaded:
                raise ImportError("TensorFlow is required for the provided ML model. Please install it to proceed.")
            if tf.config.list_physical_devices('GPU'):
                self.device = '/GPU:0'
            else:
                self.device = '/CPU:0'
            tf.device(self.device)
            if not hasattr(self.ml_model, 'optimizer'):
                self.ml_model.compile(optimizer='adam', loss='mean_squared_error')
            
    def _setup_ssh(self):
        """Configures DUV-specific settings and initializes SSH connection."""
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self._connect_ssh()
            
    def _connect_ssh(self):
        """Attempts to connect to the SSH server with retries."""
        try:
            self.ssh.connect(Config.SSH_HOST, username=Config.SSH_USERNAME)
            self.scp_client = scp.SCPClient(self.ssh.get_transport())
        except paramiko.AuthenticationException:
            print("Authentication failed, please verify your credentials.")
        except paramiko.SSHException as e:
            print(f"SSH connection failed: {e}")
        except Exception as e:
            print(f"An error occurred while connecting to SSH: {e}")
        
    def _extract_geo_params(self,polygons):
        """
        Extracts necessary details from the first polygon object for import.
        NOTE: Assumes that all polygons share these parameters. 
        """
        self.eps_in = polygons[0].eps_in # review: error control if they state mesh order since lumerical doesnt support gds mesh order...
        self.z = polygons[0].z
        self.depth = polygons[0].depth
    
    def _prefab_centering(self,polygons,gds_path):
        """Moves the Prefab ML Output back to the original device center."""
        gdsii = gdspy.GdsLibrary()
        lib = gdsii.read_gds(gds_path)
        all_cells = lib.cells.values()

        # Determine the bounding box that encloses all polygons
        xmin, ymin, xmax, ymax = float('inf'), float('inf'), -float('inf'), -float('inf')
        for cell in all_cells:
            for polygon in cell.polygons:
                bbox = polygon.get_bounding_box()
                xmin = min(xmin, bbox[0][0])
                ymin = min(ymin, bbox[0][1])
                xmax = max(xmax, bbox[1][0])
                ymax = max(ymax, bbox[1][1])

        # Calculate the center of the bounding box
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2

        bb_x, bb_y = self.calculate_bounding_box(polygons)
        # Calculate the translation needed to move the center to the origin
        dx = bb_x -center_x
        dy = bb_y -center_y

        # Apply the translation to all polygons
        for cell in all_cells:
            for polygon in cell.polygons:
                polygon.translate(dx, dy)
        gdsii.write_gds(gds_path)
        
    def _apply_gaussian_filter(self, params, kernel_size, sigma):
        kernel = self.gaussian_kernel(kernel_size, sigma)
        kernel = kernel[None, None, :]
        params = params[None, None, :]
        filtered_params = F.conv1d(params, kernel, padding=kernel_size // 2)
        return filtered_params.squeeze()
    
    @staticmethod
    def convert_to_prefab_img(eps):
        # Step 1: Normalize the matrix to range between 0 and 1
        eps_normalized = (eps - np.min(eps)) / (np.max(eps) - np.min(eps))
        
        # Step 2: Invert the grayscale values (so 0 becomes white, max becomes black)
        eps_inverted = 1 - eps_normalized
        
        # Step 3: (Optional) Convert to 0-255 range for saving as an image
        eps_grayscale = (eps_inverted * 255).astype(np.uint8)
        
        img = Image.fromarray(eps_grayscale)
        img_path = f'{Config.RESULTS_PATH}/prefab_img.gds'
        img.save(img_path)
    
        return img_path

    def convert_prefab_predict_to_eps(prediction, eps_max):
        pred_min = np.min(prediction)
        pred_max = np.max(prediction)
        normalized_pred = (prediction - pred_min) / (pred_max - pred_min)
        
        inverted_pred = 1 - normalized_pred
        eps_converted = inverted_pred * eps_max
        
        return eps_converted
    @staticmethod
    def extract_height(polygons):
        """Extracts the maximum height between bounds of all polygons. Required for deps zip method."""
        all_y_coords = [y for point in polygons for x, y in point]
        height = (max(all_y_coords) - min(all_y_coords)) * 1e9 # review: precisions etc. 
        return height
    
    @staticmethod
    def select_polygons_in_y_range(cell, layer_index, y_start, y_end):
        """Selects all polygons within the provided height range. Required for deps zip method."""
        selected_polygons = []
        shape_iterator = cell.begin_shapes_rec(layer_index)
        while not shape_iterator.at_end():
            shape = shape_iterator.shape()
            if shape.is_polygon():
                polygon = shape.polygon
                bbox = polygon.bbox()
                if bbox.bottom >= y_start and bbox.top <= y_end:
                    selected_polygons.append(polygon)
            shape_iterator.next()
        return selected_polygons
    
    @staticmethod
    def calculate_bounding_box(polygons):
        """Finds the center of the geometry."""
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        for polygon in polygons:
            xs, ys = zip(*polygon.points)  # Unzip the list of tuples into two tuples of x and y coordinates
            min_x = min(min_x, *xs)
            min_y = min(min_y, *ys)
            max_x = max(max_x, *xs)
            max_y = max(max_y, *ys)
        
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        return center_x * 1e6, center_y * 1e6
    
    @staticmethod
    def gaussian_kernel(size, sigma):
        kernel = torch.arange(size, dtype=torch.float32) - size // 2
        kernel = torch.exp(-0.5 / (sigma ** 2) * kernel ** 2)
        kernel = kernel / torch.sum(kernel)
        return kernel
     
    @staticmethod
    def recover_vertices(litho_image):
        print('not implemented yet')
    