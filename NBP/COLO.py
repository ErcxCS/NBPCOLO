import numpy as np
from sklearn.metrics import euclidean_distances
from matplotlib import pyplot as plt
from scipy.stats import norm, gaussian_kde

def mono_potential_bbox(bbox: np.ndarray):
    """
    Returns a function of prior probability for a given bounded box.

    Parameters
    --
    bbox: np.ndarray
        Bounded box for a node in 2D space.

    Returns
    --
    joint_pdf: function
        Returns a uniform pdf function for a given bounded box created for a node.
    """
    x_min, x_max, y_min, y_max = bbox
    bbox_area = (x_max - x_min) * (y_max - y_min)
    
    def joint_pdf(r: np.ndarray) -> np.ndarray:

        inside = (x_min <= r[:, 0]) & (r[:, 0] <= x_max) & (y_min <= r[:, 1]) & (r[:, 1] <= y_max)
        return inside / bbox_area

    return joint_pdf


def duo_potential(xr: np.ndarray, xu: np.ndarray, dru: int, sigma:float) -> np.ndarray:
    """
    Pair-wise potential function for particles of node r and u
    Usage: m_vu = duo_potential(M_new[u], Mu[0], D[v, u], sigma*factor)

    Parameters
    --
    xr: np.ndarray
        Particles of node r
    xu: np.ndarray
        Particle of node u
    dru: int
        Measured distance between node r and node u
    sigma: float
        Standard deviation for set of particles of node r

    Returns
    --
    likelihoods: np.ndarray
        Array of likelihoods for each particle
    """
    dist = np.linalg.norm(xr - xu, axis=1)
    return norm.pdf(dru - dist, scale=sigma)

def duo_potential_single(xr: np.ndarray, xu: np.ndarray, dru: int, sigma:float) -> np.ndarray:
    """
    Pair-wise potential function for particles of node r and u
    Usage: m_vu = duo_potential(M_new[u], Mu[0], D[v, u], sigma*factor)

    Parameters
    --
    xr: np.ndarray
        Particles of node r
    xu: np.ndarray
        Particle of node u
    dru: int
        Measured distance between node r and node u
    sigma: float
        Standard deviation for set of particles of node r

    Returns
    --
    likelihoods: np.ndarray
        Array of likelihoods for each particle
    """
    dist = np.linalg.norm(xr - xu, axis=1)
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

def neff(weights: np.ndarray):
    """ Number of effective particle points for a sample given its weights
    
    Parameters
    --
    weights: array_like
        weights of a samples's particles

    Returns
    --
    neff: int
        number of effective particles
    """
    return int(1 / sum(weights**2))

def create_bbox(D: np.ndarray, anchors: np.ndarray, limits: np.ndarray): #TODO: calculate n-hop distances
    """
    Creates intersecting bounded boxes for n samples from distances to anchor nodes

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
    intersection_bboxes = np.zeros((n_samples, 2*d))

    for i in range(n_samples):
        for j in range(n_anchors):
            if i == j:
                bboxes[i, j] = limits
                continue
            for k in range(d):
                bboxes[i, j, 2*k] = max(anchors[j, k] - D[i, j], limits[2*k])
                bboxes[i, j, 2*k+1] = min(anchors[j, k] + D[i, j], limits[2*k+1])
            
            """ if i == 2:
                xmin, xmax, ymin, ymax =  bboxes[i, j]
                plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin]) """
        
        for k in range(d):
            intersection_bboxes[i, 2*k] = np.max(bboxes[i, :, k*2], axis=0)
            intersection_bboxes[i, 2*k+1] = np.min(bboxes[i, :, k*2+1], axis=0)

        
    return intersection_bboxes

def generate_particles(intersections: np.ndarray, anchors: np.ndarray, n_particles: int):
    """
    Generates particles from intersections of bounded boxes for each target sample

    Parameters
    --
    intersections: array_like, (n_samples, d_dim*2) shaped
        each intersection is craeted with distances to each anchor node
    anchors: array_like, (n_anchors, d_dim) shaped
        known locations of anchors
    n_particles: int
        number of particles that will be generated for each sample. Anchors will have themselves as particles

    Returns
    --
    all_particles: array_like, (n_samples, n_particles, d_dim) shaped
        particles for all samples
    prior_beliefs: array_like (n_samples, n_particles) shaped
        prior beliefs about the locations of each samples' particles
    """
    # Check inputs
    assert intersections.ndim == 2 and intersections.shape[1] % 2 == 0, "intersections must be a 2D array with an even number of columns"
    assert anchors.ndim == 2 and anchors.shape[1] * 2 == intersections.shape[1], "anchors must be a 2D array with half as many columns as intersections"
    assert isinstance(n_particles, int) and n_particles > 0, "n_particles must be a positive integer"

    n_samples, d = intersections.shape
    n_anchors = anchors.shape[0]
    d //= 2

    # Initialize particles and prior beliefs
    all_particles = np.zeros((n_samples, n_particles, d))
    anchor_particles = np.repeat(anchors[:, np.newaxis, :], repeats=n_particles, axis=1)
    all_particles[:n_anchors] = anchor_particles
    prior_beliefs = np.ones((n_samples, n_particles))

    # Generate particles and calculate prior beliefs
    for i in range(n_anchors, n_samples):
        bbox = intersections[i].reshape(-1, 2)
        for j in range(d):
            all_particles[i, :, j] = np.random.uniform(bbox[j, 0], bbox[j, 1], size=n_particles)
        prior_beliefs[i] = mono_potential_bbox(intersections[i])(all_particles[i])

    return all_particles, prior_beliefs

def iterative_NBP(D: np.ndarray, X: np.ndarray, anchors: np.ndarray,
                  deployment_area: np.ndarray, n_particles: int, n_iter: int, k: int):
    """
    iterative Nonparametric Belief Propagation for Cooperative Localization task.

    Parameters
    --
    D: array_like, (n_samples, n_samples) shaped
        square measured euclidean distance matrix. Distances from each node to every other node. First n_anchors are the anchor nodes.
    X: array_like, (n_samples, d_dim) shaped
        True positions of nodes we are trying to localize. Used for comparison only. First n_anchors are the anchor nodes.
    anchors: array_like, (n_anchors, d_dim) shaped
        Nodes which we know their true locations. These are first n_anchors from X
    deployment_area: array_like, (d_dim*2,) shaped
        when deployment area is a square with length m, deployment_area is = np.array([0, m, 0, m]) = (x_min, x_max, y_min, y_max)
    n_particles: int
        number of particles for each sample in X
    n_iter: int
        number of iterations
    k: Mixture Importance Sampling parameter

    Returns
    --
    pred_X: array_like, (n_anchors:n_samples, d_dim) shaped
    Estimated locations X[n_anchors:n_samples] after n_iter iterations
    """
    n_samples, d_dim = X.shape # number of nodes in fully connected graph
    n_anchors = anchors.shape[0] # number of anchors
    n_targets = n_samples - n_anchors # number of nodes to localize
    anchor_list = list(range(n_anchors)) # first n nodes are the anchors

    target_neighbours = n_samples - n_anchors - 1 # Assumes fully connected graph, find a way for not fully connected
    new_n_particles = round(k * n_particles / target_neighbours)
    n_particles = round(new_n_particles * target_neighbours / k)

    intersections = create_bbox(D, anchors, limits=deployment_area)
    M, prior_beliefs = generate_particles(intersections, anchors, n_particles)

    """ plt.scatter(X[:, 0], X[:, 1])
    for i, bbox in enumerate(intersections):
        xmin, xmax, ymin, ymax = bbox
        plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], c='red')
        plt.annotate(f"a{i}", X[i])
    plt.show() """
    messages = np.ones((n_samples, n_samples, n_particles))
    new_messages = messages.copy()
    weights = prior_beliefs / np.sum(prior_beliefs, axis=1, keepdims=True)

    for iter in range(n_iter):
        M_new = np.ones((n_samples, n_samples, new_n_particles, 2))
        m_ru = dict()
        
        for r, Mr in enumerate(M):
            for u, Mu in enumerate(M):
                if r != u:
                    v = np.random.normal(0, 2, size=n_particles)*0
                    thetas = np.random.uniform(0, 2*np.pi, size=n_particles)
                    cos_u = (D[r, u] + v) * np.cos(thetas)
                    sin_u = (D[r, u] + v) * np.sin(thetas)
                    x_ru = Mr + np.column_stack([cos_u, sin_u])

                    qq = mono_potential_bbox(intersections[u])(x_ru)
                    

                    w_ru = weights[r] / messages[r, u]
                    w_ru /= w_ru.sum()
                    w_ru *= qq
                    
                    print(w_ru)

                    kde = gaussian_kde(x_ru.T, weights=w_ru, bw_method='silverman')
                    m_ru[r, u] = kde
                    M_new[u, r] = kde.resample(new_n_particles).T


np.random.seed(23)
n, d = 8, 2
m = 25
p = 48
r = 15
a = 4
i = 5
X = np.random.uniform(0, m, size=(n, d))

noise = np.random.normal(0, 2, size=(n, n))
noise -= np.diag(noise.diagonal())
symetric_noise = (noise + noise.T) / 2
D = euclidean_distances(X) + symetric_noise*0
iterative_NBP(D=D, X=X, anchors=X[:a], deployment_area=np.array([0, m, 0, m]), n_particles=p, n_iter=i, k=1)


                    

                
                            





