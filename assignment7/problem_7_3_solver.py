import numpy as np
from scipy.special import factorial, comb
from scipy import signal
from scipy.integrate import quad
import matplotlib.pyplot as plt
import warnings

# Suppress potential RuntimeWarning from division by zero in G(w) if it occurs
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ===============================================
# PART 1: B-spline and Dual B-spline Functions (from previous problem)
# ===============================================

def bspline0(x):
    x = np.asarray(x)
    y = np.zeros_like(x, dtype=float)
    y[np.abs(x) < 0.5] = 1.0
    y[np.abs(x) == 0.5] = 0.5
    return y

def bspline1(x):
    x = np.asarray(x)
    y = np.zeros_like(x, dtype=float)
    in_range = np.abs(x) < 1
    y[in_range] = 1 - np.abs(x[in_range])
    return y

def bspline2(x):
    x = np.asarray(x)
    y = np.zeros_like(x, dtype=float)
    x0_range = np.abs(x) < 0.5
    y[x0_range] = -x[x0_range]**2 + 3/4
    x1_range = (np.abs(x) >= 0.5) & (np.abs(x) < 1.5)
    y[x1_range] = 0.5 * (np.abs(x[x1_range]) - 1.5)**2
    return y

def bspline3(x):
    x = np.asarray(x)
    y = np.zeros_like(x, dtype=float)
    x0_range = np.abs(x) < 1
    y[x0_range] = 2/3 - x[x0_range]**2 + 0.5 * np.abs(x[x0_range])**3
    x1_range = (np.abs(x) >= 1) & (np.abs(x) < 2)
    y[x1_range] = (1/6) * (2 - np.abs(x[x1_range]))**3
    return y

def bspline4(x):
    x = np.asarray(x)
    y = np.zeros_like(x, dtype=float)
    x0_range = np.abs(x) < 0.5
    x0 = x[x0_range]
    y[x0_range] = (1/24) * (6*x0**4 - 15*x0**2 + 14.375)
    x1_range = (np.abs(x) >= 0.5) & (np.abs(x) < 1.5)
    x1_abs = np.abs(x[x1_range])
    y[x1_range] = (1/24) * (-4*x1_abs**4 + 20*x1_abs**3 - 30*x1_abs**2 + 5*x1_abs + 13.75)
    x2_range = (np.abs(x) >= 1.5) & (np.abs(x) < 2.5)
    x2_abs = np.abs(x[x2_range])
    y[x2_range] = (1/24) * (x2_abs**4 - 10*x2_abs**3 + 37.5*x2_abs**2 - 62.5*x2_abs + 39.0625)
    return y

def _powplus(x, n):
    return np.maximum(x, 0)**n

def bsplineGeneralN(x, n):
    x = np.asarray(x)
    y = np.zeros_like(x, dtype=float)
    term1 = 1 / factorial(n)
    for k in range(n + 2):
        term2 = comb(n + 1, k)
        term3 = (-1)**k
        term4 = _powplus(x - k + (n + 1) / 2, n)
        y += term1 * term2 * term3 * term4
    return y

def bsplineN(x, n):
    if n == 0: return bspline0(x)
    elif n == 1: return bspline1(x)
    elif n == 2: return bspline2(x)
    elif n == 3: return bspline3(x)
    elif n == 4: return bspline4(x)
    else: return bsplineGeneralN(x, n)

def bspline_hn(L, nn):
    g_L_n_indices = np.arange(-L, L + 1)
    g_L = bsplineN(g_L_n_indices, 2 * L)
    def G(w):
        _, h_freqz = signal.freqz(g_L, 1, worN=w)
        return np.exp(1j * L * w) * h_freqz
    def H(w):
        g_val = G(w)
        return 1. / np.real(g_val) if abs(g_val) > 1e-9 else 1e9
    hn = np.zeros_like(nn, dtype=float)
    for i, n_val in enumerate(nn):
        integrand = lambda w: H(w) * np.cos(n_val * w)
        integral_val, _ = quad(integrand, -np.pi, np.pi)
        hn[i] = integral_val / (2 * np.pi)
    return hn

# ===============================================
# PART 2: New Logic for Problem 7.3
# ===============================================

def xt(t):
    """
    Defines the signal x(t) as specified in the problem.
    This function is vectorized to work with numpy arrays.
    """
    t = np.asarray(t)
    y = np.zeros_like(t, dtype=float)
    
    # Condition 1: 0 <= t < 10
    mask1 = (t >= 0) & (t < 10)
    y[mask1] = 0.5
    
    # Condition 2: 10 <= t <= 20
    mask2 = (t >= 10) & (t <= 20)
    y[mask2] = -np.sin(np.pi * t[mask2] / 10)
    
    return y

def main():
    """
    Main function to solve the approximation problem and generate plots.
    """
    # Define the time axis for plotting
    t_plot = np.linspace(-20, 40, 10000)
    
    # Create the plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    L_values = [1, 2, 3, 4]

    print("Starting B-spline approximation...")

    for i, L in enumerate(L_values):
        print(f"\nProcessing for L={L}...")
        
        # Define the range for integration and for the shifts 'n'
        # These values are taken from the MATLAB code
        P1 = -3  # Lower integration limit
        P2 = 23  # Upper integration limit
        n_range_beta = np.arange(P1, P2 + 1)
        
        # 1. Calculate the inner products beta[n] = <x(t), b_L(t-n)>
        print("  Step 1: Calculating inner products with b_L(t-n)...")
        beta = np.zeros_like(n_range_beta, dtype=float)
        for j, n in enumerate(n_range_beta):
            integrand = lambda t: xt(t) * bsplineN(t - n, L)
            # Integrate over a fixed range where the product is non-zero
            integral_val, _ = quad(integrand, P1, P2)
            beta[j] = integral_val

        # 2. Find projection coefficients alpha[n] by convolving with h_L[n]
        print("  Step 2: Convolving with h_L[n] to find projection coefficients...")
        h_coeffs = bspline_hn(L, np.arange(-20, 21))
        alpha = np.convolve(beta, h_coeffs, mode='full')
        
        # Determine the index range for the output of the convolution
        N1 = P1 - 20
        N2 = P2 + 20
        n_range_alpha = np.arange(N1, N2 + 1)
        
        # 3. Reconstruct the signal approximation x_hat(t)
        print("  Step 3: Reconstructing the signal x_hat(t)...")
        x_hat = np.zeros_like(t_plot)
        for k, n_k in enumerate(n_range_alpha):
            x_hat += alpha[k] * bsplineN(t_plot - n_k, L)
            
        # Plotting
        ax = axes[i]
        ax.plot(t_plot, xt(t_plot), 'r', label='$x(t)$')
        ax.plot(t_plot, x_hat, 'b', label=r'$\hat{x}(t)$')
        ax.set_title(f'({chr(97+i)}) $L = {L}$') # (a), (b), (c), (d)
        ax.set_xlabel('$t$')
        ax.legend()
        ax.grid(True)

    fig.suptitle('Plots of $x(t)$ and its spline approximation $\hat{x}(t)$', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure to a file
    output_filename = 'spline_approximation_plots.png'
    fig.savefig(output_filename)
    print(f"\nSuccessfully saved approximation plots to '{output_filename}'")

# Script entry point
if __name__ == "__main__":
    main()
