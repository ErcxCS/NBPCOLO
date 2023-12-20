import numpy as np
from sklearn.metrics import euclidean_distances
from matplotlib import pyplot as plt
from scipy.stats import norm, gaussian_kde

def mono_potential_bbox(bbox: np.ndarray): #TODO: Generalize for n dimension
    """
    returns a function of prior probability for a given bounded box

    Parameters
    --
    bbox: array_like
        bounded box for a node in 2d

    Returns
    --
    uni_pdf: float
        returns a uniform pdf function for a given bounded box created for a node 
    """
    x_min, x_max, y_min, y_max = bbox
    def joint_pdf(r: np.ndarray) -> float: #TODO: Generalize for all particles
        if x_min <= r[0] <= x_max and y_min <= r[1] <= y_max:
            return 1 / ((x_max - x_min) * (y_max - y_min))
        else:
            return 0
    return joint_pdf

def duo_potential(xr: np.ndarray, xu: np.ndarray, dru: int, sigma:float) -> float: #TODO: Generalize for all particles
    """
    Pair-wise potential function for particles of node r and u

    Parameters
    --
    xr:
        particle of node r
    xu:
        particle of node u
    dru:
        measured distance between node r and node u
    sigma:
        standard deviation for set of particles of node r?

    Returns
    --
    likelihood

    """
    dist = np.linalg.norm(xr - xu)
    return norm.pdf(dru - dist, scale=sigma)

def silverman_factor(neff: int, d:int):
    """
    Computes silverman factor given n effective points and dimension d
    
    Returns
    -------
    s: float
        The silverman factor.
    """
    return np.power(neff * (d + 2.0)/4.0, -1./(d + 4))

def create_bbox(D: np.ndarray, anchors: np.ndarray, limits: np.ndarray):
    """
    Creates bounded boxes for n samples from distances to anchor nodes

    Parameters
    --
    D: array_like
        (n_samples, n_samples) Measured distance matrix between samples. First len(anchors) are distances of anchors to all others
    anchors: array_like
        (n_samples, d) shaped known locations of anchors
    limits: array_like: [dx_min, dx_max, dy_min, dy_max]
        limits of the bounded boxes for all samples.
        Represents deployment area when limits.shape[0] == 1, will be applied for all samples.
        if anchors.shape[1] == 3, each bbox will be represented with 8 elements, representing limits in 3D
    
    Returns
    --
    bbox: array_like
        bounded box for n_samples
    """
    n_samples = D.shape[0]
    n_anchors, d = anchors.shape
    bboxes = np.zeros((n_samples, n_anchors, 2*d))

    for i in range(n_samples):
        for j in range(n_anchors):
            if i == j:
                continue

            for k in range(d):
                bboxes[i, j, 2*k] = max(anchors[j, k] - D[i, j], limits[2*k])
                bboxes[i, j, 2*k+1] = min(anchors[j, k] + D[i, j], limits[2*k+1])

    return bboxes


def find_intersecting_bbox(D: np.ndarray, anchors: np.ndarray, limits: np.ndarray):
    """
    Finds the intersections of the bounding boxes created for n samples from distances to anchor nodes

    Parameters
    --
    D: array_like
        (n_samples, n_samples) Measured distance matrix between samples. First len(anchors) are distances of anchors to all others
    anchors: array_like
        (n_samples, d) shaped known locations of anchors
    limits: array_like: [dx_min, dx_max, dy_min, dy_max]
        limits of the bounded boxes for all samples.
        Represents deployment area when limits.shape[0] == 1, will be applied for all samples.
        if anchors.shape[1] == 3, each bbox will be represented with 8 elements, representing limits in 3D
    
    Returns
    --
    intersections: array_like
        intersections of the bounding boxes for n_samples
    """
    anchor_bboxes = create_bbox(D, anchors, limits)
    n_samples, n_anchors, _ = anchor_bboxes.shape
    d = anchors.shape[1]
    intersections = np.zeros((n_samples, 2*d))

    for i in range(n_samples):
        for k in range(d):
            intersections[i, 2*k] = np.max(anchor_bboxes[i, :, 2*k], axis=0)
            intersections[i, 2*k+1] = np.min(anchor_bboxes[i, :, 2*k+1], axis=0)

    return intersections


    


                    

                
                            





