import numpy as np
from sklearn.metrics import euclidean_distances
from matplotlib import pyplot as plt
from scipy.stats import norm, gaussian_kde


def rmse(X, X_hat_ab):
    # X is the true coordinates matrix
    # X_hat_ab is the estimated coordinates matrix

    # Compute the Euclidean distance for each node
    error = np.sqrt(np.sum((X - X_hat_ab)**2, axis=1))
    error2 = np.linalg.norm(X - X_hat_ab)
    # Compute the RMSE
    return np.sqrt(np.mean(error**2))

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
            if i == j or D[i, j] == 0:
                bboxes[i, j] = limits
                continue
            for k in range(d):
                bboxes[i, j, 2*k] = max(anchors[j, k] - D[i, j], limits[2*k])
                bboxes[i, j, 2*k+1] = min(anchors[j, k] + D[i, j], limits[2*k+1])
        
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

def plot_preds(X_true: np.ndarray, X_preds: np.ndarray, anchors: np.ndarray):
    n_targets, d = X_true.shape
    n_anchors = anchors.shape[0]
    
    plt.scatter(X_true[:, 0], X_true[:, 1], label="true_")

def detection_probability(Xt: np.ndarray, xu: np.ndarray, radius: int):
    return np.exp(np.linalg.norm(Xt - xu, axis=1)**2 / (-2 * radius**2))

import open3d as o3d

def create_point_cloud(data):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    return pcd

def display_point_cloud(pcd):
    o3d.visualization.draw_geometries([pcd])

def create_anchor_cloud(anchor_ids):
    data = np.random.rand(len(anchor_ids), 3) # Generate random points
    pcd = create_point_cloud(data)
    pcd.colors = o3d.utility.Vector3dVector(np.random.rand(len(anchor_ids), 3)) # Assign random colors
    return pcd


def iterative_NBP(D: np.ndarray, X: np.ndarray, anchors: np.ndarray,
                  deployment_area: np.ndarray, n_particles: int, n_iter: int, k: int, radius: int):
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
    n_samples = D.shape[0] # number of nodes in the graph
    n_anchors, d_dim = anchors.shape # number of anchors
    n_targets = n_samples - n_anchors # number of nodes to localize
    anchor_list = list(range(n_anchors)) # first n nodes are the anchors
    
    intersections = create_bbox(D, anchors, limits=deployment_area)
    M, prior_beliefs = generate_particles(intersections, anchors, n_particles)

    messages = np.ones((n_samples, n_samples, n_particles))
    
    new_messages = np.ones((n_samples, n_samples, k*n_particles))
    weights = prior_beliefs / np.sum(prior_beliefs, axis=1, keepdims=True)

    _rmse = []
    
    # we have different prior beliefs for each node, however we do not use them
    # anywhere, we normalize weights, because all priors are the same, all weights
    # for all particles of all nodes will be the same.
    
    for iter in range(n_iter):
        ############################ PLOTING #############################################
        weighted_means = np.einsum('ijk,ij->ik', M[n_anchors:], weights[n_anchors:])
        plt.scatter(anchors[:, 0], anchors[:, 1], label="anchors", c="r", marker='*') # anchors nodes
        plt.scatter(X[n_anchors:, 0], X[n_anchors:, 1], label="true", c="g", marker='P') # target nodes
        plt.scatter(weighted_means[:, 0], weighted_means[:, 1], label="preds", c="y", marker="X") # preds
        plt.plot([X[n_anchors:, 0], weighted_means[:, 0]], [X[n_anchors:, 1], weighted_means[:, 1]], "k--")
        for i, xt in enumerate(X):

            if i < n_anchors:
                plt.annotate(f"A_{i}", xt)
            else:
                bbox = intersections[i]
                xmin, xmax, ymin, ymax = bbox
                plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin])
                plt.scatter(M[i, :, 0], M[i, :, 1], marker=".", s=10)
                plt.annotate(f"t_{i}", xt)
                plt.annotate(f"p_{i}", weighted_means[i - n_anchors])
        plt.legend()
        plt.title(f"iter: {iter}, Predictions with initial weights")
    

        plt.show()
        ################################################################################
        m_ru = dict()
        M_new = [[] for i in range(n_samples) if i not in anchor_list]
        
        for r, Mr in enumerate(M):
            # sender
            #plt.scatter(Mr[:, 0], Mr[:, 1], label=f"particles of {r}")
            for u, Mu in enumerate(M): # skip anchors?
                # receiver
                if u in anchor_list:
                    continue
                if r != u and D[r, u] != 0 and D[r, u] < radius:
                    v = np.random.normal(0, 2, size=n_particles)*0
                    thetas = np.random.uniform(0, 2*np.pi, size=n_particles)
                    cos_u = (D[r, u] + v) * np.cos(thetas)
                    sin_u = (D[r, u] + v) * np.sin(thetas)
                    x_ru = Mr + np.column_stack([cos_u, sin_u])             
                    
                    pd = detection_probability(x_ru, weighted_means[n_anchors - u], radius)

                    w_ru = pd * (weights[r] / messages[r, u])
                    w_ru /= w_ru.sum()

                    """ plt.scatter(X[u, 0], X[u, 1], label="node u", c="r", marker='*')
                    plt.scatter(X[r, 0], X[r, 1], label="node r", c="g", marker="+")
                    plt.annotate(f"target {u}", X[u])

                    plt.scatter(x_ru[:, 0], x_ru[:, 1], marker=".", s=10)
                    for j, p in enumerate(x_ru):
                        plt.annotate("{:.2f}".format(w_ru[j]), p) """

                    kde = gaussian_kde(x_ru.T, weights=w_ru, bw_method='silverman')
                    m_ru[r, u] = kde
                    # kde constructed with particles of r to evaluate resampled particles of u

                    mask = (D[u] > 0) & (D[u] < radius)
                    neigbour_count = np.count_nonzero(mask)
                    new_n_particles = (n_particles*k) // neigbour_count
                    sampled_particles = kde.resample(new_n_particles).T
                    M_new[n_anchors - u].append(sampled_particles)
                    # We still need x_ru for anchors because they send messages
                #plt.show()
        for qq, Mu_new in enumerate(M_new):
            print("Test: ", len(Mu_new))
            if len(Mu_new) != 0:
                M_new[qq] = np.concatenate(Mu_new)

        for u, Mu_new in enumerate(M_new): # skip anchors?
            # receiver
            u += n_anchors

            q = []
            p = []
            for v, Mv in enumerate(M): # if has no neighbour?
                # sender
                if u != v and D[u, v] != 0:
                    if D[u, v] < radius:
                        m_vu = m_ru[v, u](Mu_new.T) # Mu_new is the sampled particles from u's neighbour kde's
                        q.append(m_vu)
                    else:
                        if v in anchor_list:
                            pd = detection_probability(Mu_new, anchors[v], radius)
                        else:
                            pd = np.array([np.sum(weights[v] * detection_probability(Mv, xu, radius)) for xu in Mu_new])
                        m_vu = 1 - pd
                    # m_vu is messages of particles from v to u
                    # m_ru[v, u] is the kde constructed with particles of v (r == v) with distance to u
                    p.append(m_vu)
                    #new_messages[u, v] = m_vu # message that u receives from v
        
            proposal_product = np.prod(p, axis=0)
            proposal_sum = np.sum(q, axis=0)

            W_u = proposal_product / proposal_sum
            W_u /= W_u.sum()
            print("p: ", proposal_product.shape)
            print("q: ", proposal_sum.shape)
            print("wu: ", W_u.shape)
            print("Mu_new: ", Mu_new.shape)

            mask = (D[u] > 0) & (D[u] < radius)
            neigbour_count = np.count_nonzero(mask)
            new_n_particles = (n_particles*k) // neigbour_count
            print("neighbours: ", neigbour_count)
            print("particles: ", new_n_particles)

            idxs = np.arange(new_n_particles * neigbour_count)
            print(new_n_particles * neigbour_count)
            indices = np.random.choice(idxs, size=n_particles, replace=True, p=W_u)

            M[u] = Mu_new[indices]
            weights[u] = W_u[indices] #???
            weights[u] /= weights[u].sum()
            messages[u] = new_messages[u, :, indices].T
            

        weighted_means = np.einsum('ijk,ij->ik', M[n_anchors:], weights[n_anchors:])
        _rmse.append(rmse(X[n_anchors:], weighted_means))
        if (iter + 1) % 10 == 0:
            plt.plot(np.arange(iter + 1),  _rmse)
            """ plt.scatter(X[n_anchors:, 0], X[n_anchors:, 1], color='c')
            plt.scatter(X[:n_anchors, 0], X[:n_anchors, 1], color='m')
            plt.scatter(weighted_means[:, 0], weighted_means[:, 1], color='y') """
            """ for i, bbox in enumerate(intersections):
                if i >= n_anchors:
                    xmin, xmax, ymin, ymax = bbox
                    plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], label=f"{i}")
                    plt.annotate(f"Xt{i}", X[i])
                    plt.annotate(f"Pt{i}", weighted_means[i])
                else:
                    plt.annotate(f"A{i}", X[i])
                    plt.annotate(f"Pa{i}", weighted_means[i]) """
            plt.legend()
            plt.show()

        
np.random.seed(42)
n, d = 16, 2
m = 50
p = 30
r = 20
a = 5
i = 10
k = 2
area = np.array([0, m, 0, m])
X =  np.empty((n, d))
bbox = area.reshape(-1, 2)
for j in range(d):
    X[:, j] = np.random.uniform(bbox[j, 0], bbox[j, 1], size=n)


noise = np.random.normal(0, 2, size=(n, n))
noise -= np.diag(noise.diagonal())
symetric_noise = (noise + noise.T) / 2
D = euclidean_distances(X) + symetric_noise*0

import networkx as nx

DD, D = D.copy(), D.copy()
D[D > r] = 0

# Fully connected
G = nx.from_numpy_array(D)

# Initialize a zero matrix for two_hop
two_hop = np.zeros((n, n))

# Calculate the shortest path lengths for all pairs of nodes in G
for j, paths in nx.all_pairs_shortest_path(G, 2):
    for q, _ in paths.items():
        two_hop[j,q] = nx.shortest_path_length(G, j, q, weight='weight')

# Create a subplot with 3 rows and 1 column
fig, axs = plt.subplots(3, 1, figsize=(6, 18))

# Create graphs from the numpy arrays
GGG, G, GG = nx.from_numpy_array(DD), nx.from_numpy_array(D), nx.from_numpy_array(two_hop)

# Titles for each subplot
titles = ["Fully Connected", "1-hop", "2-hop"]

# Draw the edges, labels, and scatter plot for each graph on its respective subplot
for graph, ax, title in zip([GGG, G, GG], axs, titles):
    nx.draw_networkx_edges(graph, pos=X, ax=ax, width=0.5)
    #nx.draw_networkx_labels(graph, X, ax=ax)
    #ax.scatter(X[:, 0], X[:, 1])
    ax.set_title(title)

# Show the plot
plt.show()

iterative_NBP(D=two_hop, X=X, anchors=X[:a], deployment_area=np.array([0, m, 0, m]), n_particles=p, n_iter=i, k=k, radius=r)
#print(np.exp( (-1/2 * (10**2))/ (5**2)))

                    

                
                            





