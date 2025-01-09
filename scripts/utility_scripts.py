"""
utility_scripts.py

Contains various utility functions
"""
import time
from config import Config
from lumopt_gl.utilities.lithography import LithographyOptimizer
import copy
import pya

def gds_extraction(geometry, lithography_model = None, file_name = None, gds_layer = 1, gds_litho_data_type = 69):
    """
    Writes geometry to a GDS file.

    Parameters:
        geometry (list): A list of polygons
        lithography_model (str): if provided, the GDS lithography model will also be saved.
        file_name (str): file names for the saved GDS. (optional)
        gds_layer (int): GDS layer for the file (optional)
        gds_litho_data_type (int): GDS datatype (optional)
    
    Returns:
        gds_path (str): path of the resulting GDS file.
    """
    layout = pya.Layout()
    top_cell = layout.create_cell("Top")
    for polygon in geometry:
        klayout_points = [pya.Point(x * 1e9, y * 1e9) for x, y in polygon.points]
        polygon = pya.Polygon(klayout_points)
        top_cell.shapes(layout.layer(gds_layer, gds_litho_data_type)).insert(polygon)

    gds_path = f'{Config.RESULTS_PATH}/input.gds'
    layout.write(gds_path)
    
    return gds_path

def lithography_visualizer(opts, **kwargs):
    """
    Takes a litho optimized object and an unoptimized object and produces the relevant graphs.

    Parameters:
        opt (object): an optimization object that has NOT been initialized.
        lithography_model (str): lithography model to test. 
    
    **kwargs:
            - stop_when_success (bool): a 'success' is defined as when the litho-optimized geometry
                                        performs better than the litho output of the unoptimized geometry.
            - time (bool): flag to return time required to optimize

    Returns:
        list: A list of results from each optimization run.
    """
    results = []
    durations = []
    for item in optimization_configs:
        # Check if the item is a tuple with configuration or just an optimization object
        if isinstance(item, tuple):
            opt, config = item
            if 'method' in config:
                opt.optimizer.method = config['method']
            if 'max_iter' in config:
                opt.optimizer.max_iter = config['max_iter']
            if 'lithography_model' in config:
                opt.geometry.lithography_model = config['lithography_model']
        else:
            opt = item

        # Run the optimization
        start_time = time.time()
        result = opt.run()
        end_time = time.time()
        
        results.append(result)
        durations.append(end_time - start_time)

    return [results, durations]
