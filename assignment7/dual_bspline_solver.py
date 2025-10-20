import numpy as np
from scipy.special import factorial, comb
from scipy import signal
from scipy.integrate import quad
import matplotlib.pyplot as plt
import warnings

# Suppress potential RuntimeWarning from division by zero in G(w) if it occurs
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ===============================================
# B-spline Function Definitions
# ===============================================

def bspline0(x):
    """B-spline function of 0th order."""
    x = np.asarray(x)
    y = np.zeros_like(x, dtype=float)
    y[np.abs(x) < 0.5] = 1.0
    y[np.abs(x) == 0.5] = 0.5 # Definition at the edge
    return y

def bspline1(x):
    """B-spline function of 1st order."""
    x = np.asarray(x)
    y = np.zeros_like(x, dtype=float)
    in_range = np.abs(x) < 1
    y[in_range] = 1 - np.abs(x[in_range])
    return y

def bspline2(x):
    """B-spline function of 2nd order."""
    x = np.asarray(x)
    y = np.zeros_like(x, dtype=float)
    x0_range = np.abs(x) < 0.5
    y[x0_range] = -x[x0_range]**2 + 3/4
    x1_range = (np.abs(x) >= 0.5) & (np.abs(x) < 1.5)
    y[x1_range] = 0.5 * (np.abs(x[x1_range]) - 1.5)**2
    return y

def bspline3(x):
    """B-spline function of 3rd order."""
    x = np.asarray(x)
    y = np.zeros_like(x, dtype=float)
    x0_range = np.abs(x) < 1
    y[x0_range] = 2/3 - x[x0_range]**2 + 0.5 * np.abs(x[x0_range])**3
    x1_range = (np.abs(x) >= 1) & (np.abs(x) < 2)
    y[x1_range] = (1/6) * (2 - np.abs(x[x1_range]))**3
    return y

def bspline4(x):
    """B-spline function of 4th order."""
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
    """Helper function: computes max(x, 0)^n."""
    return np.maximum(x, 0)**n

def bsplineGeneralN(x, n):
    """B-spline function of order n (general but inefficient implementation)."""
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
    """Dispatcher function for B-splines of order n."""
    if n == 0: return bspline0(x)
    elif n == 1: return bspline1(x)
    elif n == 2: return bspline2(x)
    elif n == 3: return bspline3(x)
    elif n == 4: return bspline4(x)
    else: return bsplineGeneralN(x, n)

# ===============================================
# Dual B-spline Coefficient Function
# ===============================================

def bspline_hn(L, nn):
    """Computes the coefficients h_L[n] for the dual B-spline function."""
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
# Main Script Logic
# ===============================================
def main():
    """Main function to run the computation and generate plots."""
    Q = 20
    nn = np.arange(-Q, Q + 1)
    L_values = [1, 2, 3, 4]

    # --- Part 1: Compute and plot h_L[n] ---
    print("Computing and plotting h_L[n]...")
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))
    axes1 = axes1.flatten()
    all_h_coeffs = {}

    for i, L in enumerate(L_values):
        print(f"  Calculating h_L[n] for L={L}...")
        h = bspline_hn(L, nn)
        all_h_coeffs[L] = h
        ax = axes1[i]
        ax.stem(nn, h, basefmt=" ", markerfmt='o')
        ax.set_title(f'$h_{L}[n]$ for L={L}')
        ax.set_xlabel('$n$')
        ax.set_ylabel(f'$h_{L}[n]$')
        ax.grid(True)
        ax.set_xlim([-20, 20])

    fig1.suptitle('Plots of $h_L[n]$ for $L \in \{1, 2, 3, 4\}$', fontsize=16)
    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig1_filename = 'h_L_n_plots.png'
    fig1.savefig(fig1_filename)
    print(f"\nSaved stem plots to '{fig1_filename}'")
    
    # --- Part 2: Compute and plot the dual B-spline functions ---
    print("\nComputing and plotting dual B-spline functions...")
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    axes2 = axes2.flatten()
    t = np.linspace(-10, 10, 20000)

    for i, L in enumerate(L_values):
        print(f"  Calculating tilde_b_L(t) for L={L}...")
        h = all_h_coeffs[L]
        bLd = np.zeros_like(t)
        for k, n_k in enumerate(nn):
            bLd += h[k] * bsplineN(t - n_k, L)
        ax = axes2[i]
        ax.plot(t, bLd)
        ax.set_title(fr'$\tilde{{b}}_{L}(t)$ for L={L}')
        ax.set_xlabel('$t$')
        ax.set_ylabel(fr'$\tilde{{b}}_{L}(t)$')
        ax.grid(True)

    fig2.suptitle(r'Plots of the dual B-spline functions $\tilde{b}_L(t)$', fontsize=16)
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig2_filename = 'dual_bspline_plots.png'
    fig2.savefig(fig2_filename)
    print(f"Saved dual B-spline plots to '{fig2_filename}'")

# ===============================================
# Script Entry Point - This part was missing
# ===============================================
if __name__ == "__main__":
    main()