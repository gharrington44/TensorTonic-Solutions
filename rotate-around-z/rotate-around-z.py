import numpy as np

def rotate_around_z(points, theta):
    """
    Rotate 3D point(s) around the Z-axis by angle theta (radians).
    """
    # Your code here
    points = np.array(points)
    
    single_point = points.ndim == 1
    
    if single_point:
        points = points.reshape(1, 3)
    
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])
    
    rotated_points = points @ rotation_matrix.T
    
    if single_point:
        rotated_points = rotated_points.reshape(3,)

    return rotated_points
    