import numpy as np
import matplotlib.pyplot as plt
from haar_transform import haar, ihaar # Import functions from the previous file

def solve_and_plot(filename, L=3):
    """
    Loads data from a CSV file, performs Haar transform, verifies 
    energy preservation, and saves the plot to a file.
    
    Args:
    filename (str): The name of the .csv file (e.g., 'blocks.csv').
    L (int): The number of levels for the Haar transform.
    """
    # 1. Load data from CSV
    print(f"--- Processing signal: {filename} ---")
    x = np.loadtxt(filename)
    signal_name = filename.split('.')[0]
    
    # 2. Perform Haar transform
    w = haar(x, L)
    
    # (Optional) Test if the inverse transform can reconstruct perfectly
    x_reconstructed = ihaar(w, L)
    reconstruction_error = np.linalg.norm(x - x_reconstructed)
    print(f"Reconstruction Error (||x - ihaar(haar(x))||): {reconstruction_error:.4f}")

    # 3. Verify energy preservation
    norm_x = np.linalg.norm(x)
    norm_w = np.linalg.norm(w)
    print(f"Norm of the original signal ||x||: {norm_x:.2f}")
    print(f"Norm of the transform coefficients ||w||: {norm_w:.2f}")
    if np.isclose(norm_x, norm_w):
        print("Energy preservation verified!")
    else:
        print("Warning: Energy not preserved.")

    # 4. Plotting
    # The signal length is 2^J. For these signals, J=10, so length is 1024.
    N = len(x)
    # After L=3 levels, the coarsest scale coeffs have a length of N / (2^L) = 1024 / 8 = 128.
    scale_len = N // (2**L)

    plt.figure(figsize=(10, 8))
    
    # Plot the original signal x[n]
    plt.subplot(5, 1, 1)
    plt.stem(x, markerfmt=' ', basefmt=' ')
    plt.title(f'x[n] ({signal_name})')
    
    # Plot the scale coefficients at scale J - 3
    plt.subplot(5, 1, 2)
    plt.stem(w[0:scale_len], markerfmt='o', basefmt='C0-')
    plt.title('Scale coefficients, j = J - 3')
    
    # Plot the wavelet coefficients at scale J - 3
    plt.subplot(5, 1, 3)
    plt.stem(w[scale_len : 2*scale_len], markerfmt='o', basefmt='C0-')
    plt.title('Wavelet coefficients, j = J - 3')

    # Plot the wavelet coefficients at scale J - 2
    plt.subplot(5, 1, 4)
    plt.stem(w[2*scale_len : 4*scale_len], markerfmt='o', basefmt='C0-')
    plt.title('Wavelet coefficients, j = J - 2')

    # Plot the wavelet coefficients at scale J - 1
    plt.subplot(5, 1, 5)
    plt.stem(w[4*scale_len : 8*scale_len], markerfmt='o', basefmt='C0-')
    plt.title('Wavelet coefficients, j = J - 1')
    
    plt.tight_layout() # Adjust subplot params for a tight layout
    
    # 5. Save the plot to a file instead of showing it
    output_filename = f"{signal_name}_plot.png"
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}\n")


if __name__ == '__main__':
    solve_and_plot('blocks.csv')
    solve_and_plot('bumps.csv')