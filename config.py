import os

class Config:
    # Get the directory where this Config.py file is located
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    # If Config.py is in a subdirectory (e.g., 'config'), adjust accordingly
    # ROOT_DIR = os.path.abspath(os.path.join(script_dir, '..'))
    
    RESULTS_PATH = os.path.join(ROOT_DIR, 'results')
    
    # Path to Lumerical API
    LUMERICAL_API_PATH = 'C:/Program Files/Lumerical/v232/api/python'  # ensure v231 or greater
    
    #####
    ## If using Computational Lithography Model / A server distribution
    #####
    
    SSH_HOST = 'hydra1.ece.ubc.ca'
    SSH_USERNAME = 'name here'
    
    # Path to lithography model on the server
    MODEL_PATH = '/ubc/ece/home/nano/data/LC_GROUP/Lithography_Model_V4_Published'
    
    # Commands to execute via SSH
    SSH_COMMAND = (
        'cd /ubc/ece/home/nano/data/LC_GROUP/Lithography_Model_V4_Published; '
        'source /CMC/scripts/mentor.calibre.2023.2_35.23.csh; '
        'calibre -drc litho.svrf'
    )
    
    # Path to GDS output from lithography model
    GDS_PATH = f'{MODEL_PATH}/results.gds'
