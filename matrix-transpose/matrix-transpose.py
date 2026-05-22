import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    A = np.array(A)
    rows, cols = A.shape
    new_arr = np.zeros((cols, rows))                      

    for i in range(rows):
        for j in range(cols):
            new_arr[j][i] = A[i][j]            

    return new_arr