import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    # Write code here
    # Raise ValueError if mismatched lengths
    if len(x) != len(y):
        raise ValueError("x and y must have the same length") 
        
    # Hint 1: Convert to numpy arrays to use vectorized operations
    x = np.array(x, dtype = float)
    y = np.array(y, dtype = float)

    # Hint 2: NumPy has a built-in function for this: np.dot(x,y)
    return np.dot(x, y)
    

    