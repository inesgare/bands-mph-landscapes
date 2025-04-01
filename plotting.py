import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from confidence_bands import standard_bootstrap, multiplier_bootstrap

def set_axes_equal(ax, points):
    """Sets equal scaling for all axes."""
    x_limits = [points[:, 0].min(), points[:, 0].max()]
    y_limits = [points[:, 1].min(), points[:, 1].max()]
    z_limits = [points[:, 2].min(), points[:, 2].max()]

    # Determine the largest range
    max_range = max(x_limits[1] - x_limits[0], 
                    y_limits[1] - y_limits[0], 
                    z_limits[1] - z_limits[0]) / 2.0

    # Calculate midpoints
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    # Set new limits
    ax.set_xlim(x_middle - max_range, x_middle + max_range)
    ax.set_ylim(y_middle - max_range, y_middle + max_range)
    ax.set_zlim(z_middle - max_range, z_middle + max_range)

def plot_3d(points, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5,5), subplot_kw={'projection': '3d'})
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=10, alpha=0.6)
    ax.view_init(elev=30, azim=45)
    set_axes_equal(ax, points)

def compute_kde_values(points, bandwidth=0.1):
    """
    Computes Kernel Density Estimation (KDE) values for each point in a point cloud.

    Parameters:
    - points: (N, 3) NumPy array of 3D points.
    - bandwidth: Optional bandwidth for KDE. If None, it uses Scott's rule.

    Returns:
    - kde_values: NumPy array of KDE values for each point.
    """
    # Transpose points to match scipy's KDE input format (shape: (3, N))
    kde_values = KernelDensity(bandwidth=bandwidth).fit(points).score_samples(points)

    # Normalize
    kde_values = (kde_values - kde_values.min()) / (kde_values.max() - kde_values.min())

    
    return kde_values

def plot_3d_kde(ax, points, kde_values):
    kde_values_normalized = (kde_values - kde_values.min()) / (kde_values.max() - kde_values.min())
    sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=kde_values_normalized, cmap="viridis", s=5, alpha=0.7)
    ax.view_init(elev=30, azim=45)
    set_axes_equal(ax, points)
    return sc

def plot_confidence_band_multi(landscapes, box, type='standard', k=1, alpha = 0.05, B=1000, ax=None):
    if type=='standard':
        ln, lambda_n, un = standard_bootstrap(landscapes[:,k,:,:], alpha = alpha, B=B)
    if type=='multiplier':
        ln, lambda_n, un = multiplier_bootstrap(landscapes[:,k,:,:], alpha = alpha, B=B)
    x = np.linspace(box[0][0], box[1][0], 100)
    y = np.linspace(box[0][1], box[1][1], 100)
    X, Y = np.meshgrid(x, y)
    
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')

    # Plot mean landscape and band
    ax.plot_surface(X, Y, lambda_n, cmap='viridis', alpha=0.9, edgecolor='none')
    ax.plot_surface(X, Y, ln, color='red', alpha=0.2, edgecolor='none')
    ax.plot_surface(X, Y, un, color='blue', alpha=0.2, edgecolor='none')
    ax.set_xlabel('Radius', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.legend()
    #ax.set_zlabel('Landscape value')

    ax.view_init(elev=30, azim=30) 
    plt.show()

def plot_confidence_band_single(landscapes, sample_range, type='standard', k=1, alpha = 0.05, B=1000, ax=None):
    if type=='standard':
        ln, lambda_n, un = standard_bootstrap(landscapes[:,k,:], alpha = alpha, B=B)
    if type=='multiplier':
        ln, lambda_n, un = multiplier_bootstrap(landscapes[:,k,:], alpha = alpha, B=B)

    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)

    x = np.linspace(sample_range[0], sample_range[1], 100)
    # Plot mean landscape and bands
    ax.plot(x, lambda_n, color='green', alpha=0.9,  label='Mean Landscape')
    ax.fill_between(x, ln, un, color='lightgreen', alpha=0.5, label=f'{100*(1-alpha)}% Confidence Band')
    ax.legend()
    ax.set_xlabel('Radius', fontsize=14)
    plt.show()
