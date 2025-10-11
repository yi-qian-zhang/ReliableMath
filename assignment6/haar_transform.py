import numpy as np

def haar(x, L):
    """
    Implements the Haar wavelet transform.

    Args:
    x (np.ndarray): The input signal. Its length must be a power of 2.
    L (int): The number of levels for the transform.

    Returns:
    np.ndarray: The transformed coefficient vector w.
    """
    # Base case: if the transform level is 0, return the original signal
    if L == 0:
        return x
    
    # Recursive step
    else:
        # Calculate scaling and wavelet coefficients from the sum/difference of adjacent elements
        # x[0::2] takes all even-indexed elements (1st, 3rd, ...)
        # x[1::2] takes all odd-indexed elements (2nd, 4th, ...)
        s_j = (x[0::2] + x[1::2]) / np.sqrt(2)
        w_j = (x[0::2] - x[1::2]) / np.sqrt(2)
        
        # Apply L-1 level Haar transform to the scaling coefficients s_j 
        # and concatenate with the wavelet coefficients w_j
        return np.concatenate((haar(s_j, L - 1), w_j))

def ihaar(w, L):
    """
    Implements the inverse Haar wavelet transform.

    Args:
    w (np.ndarray): The Haar transformed coefficient vector.
    L (int): The number of levels of the transform.

    Returns:
    np.ndarray: The reconstructed original signal x.
    """
    # Base case: if the transform level is 0, return the coefficients
    if L == 0:
        return w
        
    # Recursive step
    else:
        # Calculate the length of the coefficients at the next level
        coeff_len = len(w) // 2
        
        # Separate the deeper level coefficients and the current level wavelet coefficients
        deeper_coeffs = w[:coeff_len]
        w_j = w[coeff_len:]
        
        # Apply L-1 level inverse Haar transform to get the scaling coefficients s_j
        s_j = ihaar(deeper_coeffs, L - 1)
        
        # Initialize the reconstructed signal vector
        x = np.zeros(len(w), dtype=float)
        
        # Reconstruct the signal using the inverse transform formulas
        x[0::2] = (s_j + w_j) / np.sqrt(2)
        x[1::2] = (s_j - w_j) / np.sqrt(2)
        
        return x