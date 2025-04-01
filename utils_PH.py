import numpy as np
from ripser import ripser
import gudhi.representations as gdr
import multipers as mp
import multipers.filtrations as mpf
from sklearn.neighbors import KernelDensity

def compute_single_parameter_landscape(points, maxdim=1, hom_degree=1, ks=3, resolution=100, sample_range=None):
    """Compute the single parameter landscapes of the Vietoris--Rips filtration from an input point cloud

    Parameters:
    - points: (n,d) array of n points in a d-dimensional space.
    - maxdim: maximal homological dimension to compute in Ripser.
    - hom_degree: homological dimension of the landscapes to compute.
    - ks: number of landscapes to compute.
    - resolution: resolution for the computation of the landscapes.
    - sample_range: domain [a, b] over which to compute the landscapes.

    Returns:
    - single_landscape: (ks, resolution) array containing the landscapes.
    """
    diagrams = ripser(points, maxdim=maxdim, distance_matrix=False)['dgms'][hom_degree]
    lds = gdr.Landscape(num_landscapes=ks, resolution=resolution, sample_range=sample_range)
    return lds.fit_transform([diagrams]).reshape(ks, resolution)

def compute_multiparameter_landscape(points, bandwidth=0.5, threshold_radius=None, hom_degree=1, ks=3, resolution=100, box=None):
    """Function to compute the multiparameter persistence landscape for the Vietoris--Rips-density bifiltration using a KDE 
    as function to filter the points

    Parameters:
    - points: (n,d) array of n points in a d-dimensional space.
    - function: (n,) array containing the values of a function (e.g. -density estimation) used
    for the sublevel set filtration on the second parameter.
    - maxdim: maximal homological dimension to compute in Ripser.
    - ks: number of landscapes to compute.
    - resolution: resolution for the computation of the landscapes.
    - box: domain [[lower_x, lower_y], [upper_x, upper_y]] over which to compute the landscapes

    Returns:
    - single_landscape: (ks, resolution, resolution) array containing the landscapes.
    """
    f= - KernelDensity(bandwidth=bandwidth).fit(points).score_samples(points) # minus to reverse the order
    st = mpf.RipsLowerstar(points=points, function=f, threshold_radius=threshold_radius).collapse_edges(-2).expansion(2)
    bimod = mp.module_approximation(st)
    return bimod.landscapes(degree=hom_degree, resolution=[resolution, resolution], box=box, ks=range(ks))
