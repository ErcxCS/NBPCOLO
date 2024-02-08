import numpy as np
from sklearn.metrics import euclidean_distances
from matplotlib import pyplot as plt
from scipy.stats import norm, gaussian_kde
import networkx as nx


def RMSE(targets: np.ndarray, predicts: np.ndarray):
    """
    Returns the Root Mean Squared Error (RMSE) between two sets of targets and predictions.

    Parameters
    --
    - targets : array-like of shape (n_samples, 2)
        true positions of targets
    -  predicts : array-like of shape (n_samples, 2)
        prediction of posisitions of targets

    Returns
    --
    Euclidean  distance between `predicts` and `targets`. The square root of the mean squared error
    """
    error = np.sqrt(np.sum((targets - predicts)**2, axis=1))
    rmse = np.sqrt(np.mean(error**2))
    return rmse

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

def duo_potential(X_r: np.ndarray, x_u: np.ndarray, d_ru: int, sigma:float) -> np.ndarray:
    """
    Pair-wise potential function for particles of node r and u
    Usage: m_vu = duo_potential(M_new[u], Mu[0], D[v, u], sigma*factor)

    Parameters
    --
    X_r: np.ndarray
        Particles of node r
    x_u: np.ndarray
        Particle of node u
    d_ru: int
        Measured distance between node r and node u
    sigma: float
        Standard deviation for set of particles of node r

    Returns
    --
    likelihoods: np.ndarray
        Array of likelihoods for each particle
    """
    dist = np.linalg.norm(X_r - x_u, axis=1)
    likelihood = norm.pdf(d_ru - dist, scale=sigma)
    return likelihood

def silverman_factor(neff: int, d:int) -> float:
    """
    Computes silverman factor given n effective points and dimension d
    
    Returns
    -------
    s: float
        The silverman factor.
    """
    return np.power(neff * (d + 2.0)/4.0, -1./(d + 4))

def neff(weights: np.ndarray) -> int:
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

def generate_targets(seed: int = None,
                     shape: tuple[int, int] = (50, 2),
                     deployment_area: int = 100, # write method for custom deployment area
                     n_anchors:  int = 6,
                     show = False
                     ):
    """ Generate targets randomly within the deployment area
    Parameters
    ----
    seed : int or None
        If provided will set numpy random seed to this value
        
    shape : tuple of length 2
        Shape of target matrix [n_nodes, n_dims]. Default is (50, 2) i.e., 50 nodes and 2 dimensions.
    deployment_area : float
        Deployment area in which nodes are generated. Nodes are placed randomly within this area. Default is 100m.
    communication_range : int
        Range in which other agents can communicate with a node. Default is 35m.
    n_anchors : int
        Number of anchor points used when generating targets. These are placed at regular intervals around the field
        Number of anchor points used to determine the position of each target. Default is 6.
    noise_std : float
        Standard deviation of Gaussian noise added to the coordinates of the targets. Default is 0.1
    Returns
    ---
    targets : ndarray
        Target locations as an NxD numpy array where N is the number of nodes and D  is the number of dimensions.
    """
    np.random.seed(seed)
    area = np.array([-deployment_area/2, deployment_area/2, -deployment_area/2, deployment_area/2])
    deployment_bbox = area.reshape(-1, 2)
    X = np.empty(shape)
    n, d = shape

    for j in range(d):
        X[:, j] = np.random.uniform(deployment_bbox[j, 0], deployment_bbox[j, 1], size=n)
    print(f"generated {X.shape} nodes, firs n_anchors are anchors")
    if show:
        plt.scatter(X[:n_anchors, 0], X[:n_anchors, 1], c='r', marker='*')
        plt.scatter(X[n_anchors:, 0], X[n_anchors:, 1], c='y', marker='+')
        plt.show()
    
    return X, area

def get_distance_matrix(X_true: np.ndarray, noise=.2) -> np.ndarray:
    D = euclidean_distances(X_true)
    if noise == None:
        return D
    
    high, mid, low = 20, 35, 60
    P_i = -np.linspace(10, 35, D.shape[0])
    alpha = 3.15 # path loss exponent
    d0 = 1.15
    epsilon = 1e-9
    def distance_2_RSS(P_i, D, alpha, d0, epsilon):
        s = 10 * alpha * np.log10((D + epsilon) / d0)
        return P_i - s

    def RSS_2_distance(P_i, RSS, alpha, d0, epsilon, sigma: float=.2, noise=True):
        if noise:
            noise = np.random.lognormal(0, sigma=sigma, size=RSS.shape)
            noise -= np.diag(noise.diagonal())
            symetric_noise = (noise + noise.T) / 2
            RSS += symetric_noise

        d = d0 * 10 ** ((P_i - RSS) / (10 * alpha)) + epsilon
        return d

    RSS = distance_2_RSS(P_i, D,  alpha, d0, epsilon)
    return RSS_2_distance(P_i, RSS, alpha, d0, epsilon, sigma=noise, noise=True)

def get_graphs(D: np.ndarray, communication_range) -> dict:
    graphs = {}
    graphs["full"] = D
    one_hop = D.copy()
    one_hop[one_hop > communication_range] = 0
    graphs["one"] = one_hop

    G = nx.from_numpy_array(one_hop)
    two_hop = one_hop.copy()
    for j, paths in nx.all_pairs_shortest_path(G, 2):
        for q, _ in paths.items():
            two_hop[j, q] = nx.shortest_path_length(G, j, q, weight='weight')
    
    graphs["two"] = two_hop
    return graphs

def relative_spread(particles_u: np.ndarray, particles_r: np.ndarray, d_ru: float):
    dist_ur = particles_u - particles_r
    angle_samples = np.arctan2(dist_ur[:, 1], dist_ur[:, 0])
    kde = gaussian_kde(angle_samples.T)
    samples = kde.resample(particles_u.shape[0]).T
    samples = np.mod(samples + np.pi, 2*np.pi) - np.pi
    particle_noise = np.random.normal(0, 1, size=particles_u.shape[0]) * 1
    cos_u = (d_ru + particle_noise).reshape(-1, 1) * np.cos(samples)
    sin_u = (d_ru + particle_noise).reshape(-1, 1) * np.sin(samples)
    d_xy = np.column_stack([cos_u, sin_u])
    return d_xy

def draw_particles(nbp, receiver_u: int, kde_ru: dict, n_particles: int, k: int):
    sampled_particles = []
    kn_particles = k * n_particles
    node = nbp.graph[receiver_u]
    neighbour_count = np.count_nonzero((node > 0) & (node < nbp._communication_range))
    for sender_r in  range(nbp._n_samples):
        d_ur = nbp.graph[receiver_u, sender_r]
        if receiver_u == sender_r or d_ur == 0:
            continue
        if  d_ur <= nbp._communication_range:
            new_n_particles = kn_particles // neighbour_count
            kn_particles -= new_n_particles
            neighbour_count -= 1

            drawn_samples = kde_ru[sender_r, receiver_u].resample(new_n_particles).T
            sampled_particles.append(drawn_samples)
    return np.concatenate(sampled_particles)
    

def plot_networks(X_true:np.ndarray, n_anchors: int, graphs: dict):
    fig, axs = plt.subplots(len(graphs), 1,  sharex=True, sharey=True, figsize=(6, 18))
    for (title, graph), ax in zip(graphs.items(), axs):
        ax.scatter(X_true[:n_anchors, 0], X_true[:n_anchors, 1], marker="*", c="r")
        nx.draw_networkx_edges(nx.from_numpy_array(graph), pos=X_true, ax=ax, width=0.5)
        ax.set_title(title)
    plt.show()



""" X = np.array([
    [2, 2],
    [8, 8]
])
p = 50
y = np.zeros((2, p, 2))
g = 1
for i in range(len(X)):
    y[i] = np.random.uniform(X[i, 0] - g, X[i, 0] + g, size=(p, 2))
    plt.scatter(y[i, :, 0], y[i, :, 1])
plt.scatter(X[:, 0], X[:, 1], color="red")


#angle_samples = np.arctan2(y[0, :, 1], y[0, :, 0]) - np.arctan2(y[1, :, 1], y[1, :, 0])
yy = y[1] - y[0]
angle_samples = np.arctan2(yy[:, 1], yy[:, 0])
kde = gaussian_kde(angle_samples.T)
samples = kde.resample(p).T
samples = np.mod(samples + np.pi, 2*np.pi) - np.pi
x_ru = y[0] + np.column_stack([8*+np.cos(samples), 8*np.sin(samples)])
plt.scatter(x_ru[:, 0], x_ru[:, 1])
plt.show()
 """