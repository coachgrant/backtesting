import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.linalg import block_diag
import seaborn as sns


def generate_realistic_correlation_matrix(
    n,
    n_sectors=3,
    intra_sector_corr_range=(0.5, 0.8), 
    inter_sector_corr_range=(-0.3, 0.4),
    noise_level=0.1,
    negative_corr_pairs=None
):
    """
    Generate a realistic correlation matrix with block structure representing sectors.
    
    Parameters:
    -----------
    n : int
        Size of the correlation matrix (number of assets)
    n_sectors : int
        Number of sectors/blocks
    intra_sector_corr_range : tuple
        Range for correlations within sectors (higher correlation)
    inter_sector_corr_range : tuple
        Range for correlations between sectors (can be negative)
    noise_level : float
        Amount of noise to add for realism
    negative_corr_pairs : list of tuples
        List of (sector_i, sector_j) pairs that should have negative correlation
    
    Returns:
    --------
    corr_matrix : ndarray
        NxN correlation matrix
    sector_assignments : list
        List indicating which sector each asset belongs to
    """
    # Assign assets to sectors
    assets_per_sector = n // n_sectors
    remainder = n % n_sectors
    sector_sizes = [assets_per_sector + (1 if i < remainder else 0) for i in range(n_sectors)]
    
    # Create sector assignment list
    sector_assignments = []
    for sector_id, size in enumerate(sector_sizes):
        sector_assignments.extend([sector_id] * size)
    
    # Initialize correlation matrix
    corr_matrix = np.eye(n)
    
    # Define which sector pairs should have negative correlation
    if negative_corr_pairs is None:
        # Default: sector 0 (e.g., "tech/growth") negatively correlated with sector 2 (e.g., "utilities/defensive")
        negative_corr_pairs = [(0, 2)]
    
    # Fill in correlations
    for i in range(n):
        for j in range(i+1, n):
            if sector_assignments[i] == sector_assignments[j]:
                # Same sector - higher correlation
                base_corr = np.random.uniform(*intra_sector_corr_range)
            else:
                # Different sectors
                sector_pair = tuple(sorted([sector_assignments[i], sector_assignments[j]]))
                
                if sector_pair in negative_corr_pairs:
                    # Negative correlation between these sectors
                    base_corr = np.random.uniform(-0.4, -0.1)
                else:
                    # Normal inter-sector correlation
                    base_corr = np.random.uniform(*inter_sector_corr_range)
            
            # Add noise
            noise = np.random.normal(0, noise_level)
            corr = np.clip(base_corr + noise, -0.99, 0.99)
            
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr
    
    # Ensure positive semi-definite by eigenvalue adjustment
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    corr_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    # Normalize to ensure diagonal is 1
    d = np.sqrt(np.diag(corr_matrix))
    corr_matrix = corr_matrix / np.outer(d, d)
    
    return corr_matrix, sector_assignments


def simulate_correlated_gbm(
    S0,
    mu,
    sigma,
    corr_matrix,
    T,
    dt,
    n_paths
):
    """
    Simulate correlated Geometric Brownian Motion paths.
    
    Parameters:
    -----------
    S0 : array-like
        Initial prices for each asset
    mu : array-like
        Drift parameters for each asset
    sigma : array-like
        Volatility parameters for each asset
    corr_matrix : ndarray
        Correlation matrix
    T : float
        Time horizon
    dt : float
        Time step
    n_paths : int
        Number of simulation paths
    
    Returns:
    --------
    paths : ndarray
        Array of shape (n_steps, n_assets, n_paths) containing price paths
    """
    n_assets = len(S0)
    n_steps = int(T / dt)
    
    # Cholesky decomposition for correlation
    L = np.linalg.cholesky(corr_matrix)
    
    # Initialize paths
    paths = np.zeros((n_steps + 1, n_assets, n_paths))
    paths[0, :, :] = S0[:, np.newaxis]
    
    # Generate correlated random shocks
    for t in range(1, n_steps + 1):
        # Independent standard normal random variables
        Z = np.random.standard_normal((n_assets, n_paths))
        
        # Correlate the random variables
        Z_corr = L @ Z
        
        # GBM formula
        drift = (mu[:, np.newaxis] - 0.5 * sigma[:, np.newaxis]**2) * dt
        diffusion = sigma[:, np.newaxis] * np.sqrt(dt) * Z_corr
        
        paths[t] = paths[t-1] * np.exp(drift + diffusion)
    
    return paths


def plot_correlation_matrix(
    corr_matrix,
    sector_assignments
):
    """
    Plot correlation matrix with annotations and sector boundaries.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create custom colormap (red-white-blue)
    colors = ['darkred', 'red', 'lightcoral', 'white', 'lightblue', 'blue', 'darkblue']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('correlation', colors, N=n_bins)
    
    # Plot heatmap
    im = ax.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha='center', va='center',
                          color='black' if abs(corr_matrix[i, j]) < 0.5 else 'white',
                          fontsize=8)
    
    # Add sector boundaries
    unique_sectors = sorted(set(sector_assignments))
    sector_boundaries = []
    current_pos = -0.5
    
    for sector in unique_sectors:
        sector_size = sector_assignments.count(sector)
        sector_boundaries.append(current_pos + sector_size)
        current_pos += sector_size
    
    # Draw sector boundary lines
    for boundary in sector_boundaries[:-1]:
        ax.axhline(y=boundary, color='black', linewidth=2)
        ax.axvline(x=boundary, color='black', linewidth=2)
    
    # Labels
    ax.set_xticks(range(len(corr_matrix)))
    ax.set_yticks(range(len(corr_matrix)))
    ax.set_xticklabels([f'Asset {i+1}' for i in range(len(corr_matrix))], rotation=45)
    ax.set_yticklabels([f'Asset {i+1}' for i in range(len(corr_matrix))])
    ax.set_title('Equity Correlation Matrix with Sector Structure', fontsize=14, pad=20)
    
    plt.tight_layout()
    return fig


def plot_price_paths(paths, sector_assignments, n_sample_paths=20):
    """
    Plot simulated correlated price paths.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    n_steps, n_assets, n_paths = paths.shape
    time_grid = np.arange(n_steps)
    
    # Define colors for different sectors
    sector_colors = plt.cm.tab10(np.linspace(0, 1, len(set(sector_assignments))))
    
    # Single realisation showing all assets and correlation
    for asset_idx in range(n_assets):
        sector = sector_assignments[asset_idx]
        sample_path = paths[:, asset_idx, 0]  # First simulation path
        
        ax.plot(time_grid, sample_path, 
                color=sector_colors[sector], 
                linewidth=2, 
                alpha=0.8,
                label=f'Asset {asset_idx+1} (Sector {sector+1})')
    
    ax.set_xlabel('Time Steps', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.set_title('Assets and Correlation Structure', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add legend with sector grouping
    handles, labels = ax.get_legend_handles_labels()
    sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: int(x[1].split('Sector ')[1][0]))
    handles, labels = zip(*sorted_handles_labels)
    ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Parameters
    N = 15  # Number of assets
    n_sectors = 3  # Number of sectors
    
    # Generate correlation matrix
    print("Generating realistic correlation matrix...")
    corr_matrix, sector_assignments = generate_realistic_correlation_matrix(
        N, 
        n_sectors=n_sectors,
        intra_sector_corr_range=(0.6, 0.85),
        inter_sector_corr_range=(-0.2, 0.35),  # Allow negative correlations
        noise_level=0.05,
        negative_corr_pairs=[(0, 2)]  # Sectors 0 and 2 are negatively correlated
    )
    
    # GBM parameters
    S0 = np.random.uniform(50, 150, N)  # Initial prices
    mu = np.random.uniform(0.05, 0.15, N)  # Annual drift (5-15%)
    sigma = np.random.uniform(0.15, 0.35, N)  # Annual volatility (15-35%)
    T = 1.0  # 1 year
    dt = 1/252  # Daily steps (252 trading days)
    n_paths = 1000  # Number of simulation paths
    
    # Add sector-based adjustments to parameters
    sector_names = ["Growth/Tech", "Neutral", "Defensive/Utilities"]
    for i in range(N):
        sector = sector_assignments[i]
        # Adjust parameters by sector
        if sector == 0:  # Growth/Tech sector - higher growth, higher volatility
            mu[i] *= 1.3
            sigma[i] *= 1.2
        elif sector == 2:  # Defensive/Utilities sector - lower growth, lower volatility
            mu[i] *= 0.7
            sigma[i] *= 0.6
    
    # Simulate paths
    print("Simulating correlated GBM paths...")
    paths = simulate_correlated_gbm(S0, mu, sigma, corr_matrix, T, dt, n_paths)
    
    # Plot correlation matrix
    print("Plotting correlation matrix...")
    fig_corr = plot_correlation_matrix(corr_matrix, sector_assignments)
    plt.show()
    
    # Plot price paths
    print("Plotting price paths...")
    fig_paths = plot_price_paths(paths, sector_assignments, n_sample_paths=30)
    plt.show()
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Number of assets: {N}")
    print(f"Number of sectors: {n_sectors}")
    print(f"Sector names: Growth/Tech (0), Neutral (1), Defensive/Utilities (2)")
    print(f"Assets per sector: {[sector_assignments.count(i) for i in range(n_sectors)]}")
    print(f"\nCorrelation matrix properties:")
    
    # Get upper triangle correlations (excluding diagonal)
    upper_triangle_corr = corr_matrix[np.triu_indices(N, k=1)]
    print(f"  Min correlation: {upper_triangle_corr.min():.3f}")
    print(f"  Max correlation: {upper_triangle_corr.max():.3f}")
    print(f"  Number of negative correlations: {(upper_triangle_corr < 0).sum()}")
    
    print(f"  Mean intra-sector correlation: ", end="")
    
    # Calculate mean intra-sector correlation
    intra_corr = []
    for i in range(N):
        for j in range(i+1, N):
            if sector_assignments[i] == sector_assignments[j]:
                intra_corr.append(corr_matrix[i, j])
    print(f"{np.mean(intra_corr):.3f}")
    
    print(f"  Mean inter-sector correlation: ", end="")
    # Calculate mean inter-sector correlation
    inter_corr = []
    for i in range(N):
        for j in range(i+1, N):
            if sector_assignments[i] != sector_assignments[j]:
                inter_corr.append(corr_matrix[i, j])
    print(f"{np.mean(inter_corr):.3f}")
    
    # Check positive semi-definite
    eigenvalues = np.linalg.eigvals(corr_matrix)
    print(f"\nSmallest eigenvalue: {eigenvalues.min():.6f} (should be > 0)")
    
    # Final prices statistics
    final_prices = paths[-1, :, :]
    print(f"\nFinal price statistics:")
    print(f"  Mean returns: {(final_prices.mean(axis=1) / S0 - 1).mean():.2%}")
    print(f"  Volatility of returns: {(final_prices / S0[:, np.newaxis] - 1).std(axis=1).mean():.2%}")