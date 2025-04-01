import numpy as np
import math
import matplotlib.pyplot as plt

def standard_bootstrap(landscapes, alpha = 0.05, B=1000):
    """
    Standard Bootstrap Algorithm on Persistence Landscapes to Construct Confidence Band on Single Parameter Landscapes
    Parameters:
    - landscapes: np.ndarray of shape (n, _ ) where n is the number of landscapes 
    - alpha: Confidence level (default 0.05 for 95% confidence intervals)
    - B: Number of bootstrap samples (default 1000)
    Returns: 
    - ln, un: confidence band lower and upper functions
    """
    n = landscapes.shape[0]
    
    # Step 1: Compute the average landscape λ_n(t)
    lambda_n = np.mean(landscapes, axis=0)
    
    # Steps 2 & 3: Generate bootstrap estimates of thetas
    thetas = []
    for b in range(B):
        # Resample landscapes with replacement
        resampled_indices = np.random.choice(n, math.floor(n/2), replace=True)
        resampled_landscapes = landscapes[resampled_indices]
        mean_landscape = np.mean(resampled_landscapes, axis=0)
        diff  = np.abs(np.sqrt(n)*(mean_landscape - lambda_n))
        thetas.append(np.max(diff))
    
    # Step 4: Compute quantile and upper & lower bounds
    q = np.percentile(np.array(thetas), 100*(1-alpha))
    ln = lambda_n - q / np.sqrt(n)
    un = lambda_n + q / np.sqrt(n)
    
    return ln, lambda_n, un

def multiplier_bootstrap(landscapes, alpha=0.05, B=1000, multiparameter = False):
    """
    Multiplier Bootstrap Algorithm for Persistence Landscapes
    Parameters:
    - landscapes: np.ndarray of shape (n, _ ) where n is the number of landscapes 
    - alpha: Confidence level (default 0.05 for 95% confidence intervals)
    - B: Number of bootstrap samples (default 1000)
    Returns: 
    - ln, un: confidence band lower and upper functions
    """
    n = landscapes.shape[0]
    
    # Step 1: Compute the average landscape λ_n(t) and deviation of each landscape from average
    lambda_n = np.mean(landscapes, axis=0)
    deviation = landscapes - lambda_n
    
    # Step 2: Bootstrap samples to compute 𝜃
    theta_tilde = np.zeros(B)
    for j in range(B):
        # Step 3: Generate ξ1, ..., ξn ~ N(0, 1)
        xi = np.random.normal(0, 1, n)
        
        # Step 4: Compute 𝜃_j
        if multiparameter:
            theta_tilde[j] = np.max(np.abs(np.sum(xi[:, np.newaxis, np.newaxis] * deviation, axis=0)))/ np.sqrt(n)
        else:
            theta_tilde[j] = np.max(np.abs(np.sum(xi[:, None] * deviation, axis=0)))/ np.sqrt(n)
    
    # Step 6: Compute Z̃(α)
    Z_alpha = np.percentile(theta_tilde, 100 * (1 - alpha))
    
    # Step 7: Compute confidence bands
    ln = lambda_n - Z_alpha / np.sqrt(n)
    un = lambda_n + Z_alpha / np.sqrt(n)
    
    return ln, lambda_n, un

def compute_accuracy(test_mpl, test_label, confidence_bands):
    """Compute the accuracy of a Maximum Band Depth classifier trained on the confidence bands"""
    tot_mbd_bootstrap = np.zeros((test_label.shape[0], 3))
    for j, lands in enumerate(test_mpl):
        for i in range(3):
            ln, un = confidence_bands[i]
            tot_mbd_bootstrap[j, i] = np.sum((lands >= ln) & (lands <= un))
    return np.mean(np.argmax(tot_mbd_bootstrap, axis=1) == test_label)