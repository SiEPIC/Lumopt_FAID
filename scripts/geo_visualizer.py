"""
geo_visualizer.py

Visualization scripts for polygon and geometry objects.
"""
import matplotlib.pyplot as plt

def visualize_func_polygon(function, params, title='Polygon Visualization'):
    """Generate the polygon points using the provided function and parameters"""
    
    polygon_points = function(params)
    
    # Extract x and y coordinates
    x_coords, y_coords = polygon_points[:, 0], polygon_points[:, 1]
    
    # Plot the polygon
    plt.figure(figsize=(6, 4))
    plt.plot(x_coords, y_coords, '-o', markersize=2, label='Polygon Shape')
    plt.fill(x_coords, y_coords, alpha=0.3)
    
    # Additional plot settings
    plt.title(title)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.axis('equal')
    plt.grid(False)
    plt.legend()
    plt.show()