import numpy as np

def sample_torus(R, r, num_points=1000):
    """
    Generates a sample of points from a torus in R^3 with added noise.

    Parameters:
    - R: Major radius of the torus.
    - r: Minor radius of the torus.
    - num_points: Number of points to sample.
    - noise_level: Standard deviation of Gaussian noise.

    Returns:
    - points: (num_points, 2) array of sampled points.
    """
    theta = 2 * np.pi * np.random.rand(num_points)  # Angle around the torus
    phi = 2 * np.pi * np.random.rand(num_points)    # Angle around the tube

    # Parametric equations for the torus in 3D
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)

    return np.column_stack((x, y, z))

def sample_sphere(R, num_points=1000):
    """
    Generates a sample of points from a circle in R^3 with added noise.

    Parameters:
    - R: Radius of the sphere.
    - num_points: Number of points to sample.
    - noise_level: Standard deviation of Gaussian noise.

    Returns:
    - points: (num_points, 2) array of sampled points.
    """
    theta = 2 * np.pi * np.random.rand(num_points)  # Azimuthal angle
    phi = np.arccos(2 * np.random.rand(num_points) - 1)  # Polar angle

    # Parametric equations for the sphere in 3D
    x = R * np.sin(phi) * np.cos(theta)
    y = R * np.sin(phi) * np.sin(theta)
    z = R * np.cos(phi)

    return np.column_stack((x, y, z))

def sample_klein_bottle(num_points=1000):
    """
    Generates a sample of points from a Klein bottle in R^3.

    Parameters:
    - num_points: Number of points to sample.

    Returns:
    - points: (num_points, 3) array of sampled points.
    """
    u = 2 * np.pi * np.random.rand(num_points)  # u in [0, 2pi]
    v = 2 * np.pi * np.random.rand(num_points)  # v in [0, 2pi]

    # Parametric equations for the Klein bottle
    x = (2 + np.cos(u / 2) * np.sin(v) - np.sin(u / 2) * np.sin(2 * v)) * np.cos(u)
    y = (2 + np.cos(u / 2) * np.sin(v) - np.sin(u / 2) * np.sin(2 * v)) * np.sin(u)
    z = np.sin(u / 2) * np.sin(v) + np.cos(u / 2) * np.sin(2 * v)

    return np.column_stack((x, y, z))

def normalize_point_cloud(points):
    """
    Normalize a point cloud to fit inside a unit cube (box with side length 1).

    Parameters:
    - points: (N, 3) NumPy array of 3D points.

    Returns:
    - normalized_points: (N, 3) Normalized points inside the unit cube.
    """
    # Find the min and max values for each axis (X, Y, Z)
    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)
    
    # Compute the ranges (max - min) for each axis
    ranges = max_vals - min_vals
    
    # Compute the scaling factor to fit inside a unit cube
    scale = np.max(ranges)
    
    # Scale the points to fit in the unit cube
    scaled_points = (points - min_vals) / scale
    
    # Optionally, translate the points to center the cloud inside the unit cube
    # This centers the normalized point cloud
    center_translation = 0.5 - np.mean(scaled_points, axis=0)
    normalized_points = scaled_points + center_translation
    
    return normalized_points

def add_gaussian_noise(points, noise_level=0.05):
    """
    Adds Gaussian (normal) noise to a set of 3D points.
    """
    noise = np.random.normal(0, noise_level, points.shape)
    return points + noise

def add_salt_and_pepper_noise(points, fraction=0.05, amount=0.2):
    """
    Adds salt-and-pepper noise to a set of 3D points.
    
    Parameters:
    - fraction: Proportion of points to modify.
    - amount: Intensity of the change.
    """
    noisy_points = points.copy()
    num_noisy = int(fraction * len(points))
    indices = np.random.choice(len(points), num_noisy, replace=False)

    # Randomly flip some points by a large amount
    noisy_points[indices] += np.random.choice([-amount, amount], size=(num_noisy, 3))
    
    return noisy_points