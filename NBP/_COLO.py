import numpy as np
from sklearn.metrics import euclidean_distances
from matplotlib import pyplot as plt
from scipy.stats import norm, gaussian_kde
import networkx as nx
import seaborn as sns

import numpy as np
from sklearn.metrics import euclidean_distances
from matplotlib import pyplot as plt
from scipy.stats import norm, gaussian_kde, uniform
from scipy.spatial import procrustes
import seaborn as sns
import pandas
from MDS import ClassicMDS
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import os

from COLO import detection_probability

from scipy.stats import chi2

def RMSE(targets: np.ndarray, predicts: np.ndarray):
    error = ERR(targets, predicts)
    mse = np.mean(error**2)
    rmse = np.sqrt(mse)

    mean_error = np.mean(error)
    median_error = np.median(error)
    """ mae_error = MAE(targets, predicts)
    print(f"Mean Error: {mean_error}, Median Error: {median_error}, MAE: {mae_error}")

    sns.histplot(error, kde=True, label='1-hop Errors')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.show() """
    return rmse, median_error

def ERR(y_true: np.ndarray, y_pred: np.ndarray)->float:
    return np.sqrt(np.sum((y_true - y_pred)**2, axis=1))

def MAE(y_true, y_pred):
    return np.mean(ERR(y_true, y_pred))

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
    """ c_area = area.copy()
    area[0] = -50
    area[1] = -15
    area[2] = -50
    area[3] = -15 """
    deployment_bbox = area.reshape(-1, 2)
    X = np.empty(shape)
    n, d = shape

    for j in range(d):
        X[:, j] = np.random.uniform(deployment_bbox[j, 0], deployment_bbox[j, 1], size=n)

    """ area = np.array([-deployment_area/2, deployment_area/2, -deployment_area/2, deployment_area/2])
    area[0] = 15
    area[1] = 50
    area[2] = 15
    area[3] = 50
    deployment_bbox = area.reshape(-1, 2)
    n, d = shape

    for j in range(d):
        X[n//2:, j] = np.random.uniform(deployment_bbox[j, 0], deployment_bbox[j, 1], size=n//2) """
    print(f"generated {X.shape} nodes, firs n_anchors are anchors")
    if show:
        plt.scatter(X[:n_anchors, 0], X[:n_anchors, 1], c='r', marker='*')
        plt.scatter(X[n_anchors:, 0], X[n_anchors:, 1], c='y', marker='+')
        plt.show()
    
    return X, area

def get_distance_matrix(X_true: np.ndarray, n_anchors: int, communication_radius: float, noise: float = 0.2, alpha: float = 3.15, d0: float = 1.15) -> np.ndarray:
    D = euclidean_distances(X_true)
    anchors = D[:n_anchors, :n_anchors].copy()
    D[D > communication_radius] = 0
    B = D > 0
    if noise is None:
        return D, B
    
    P_i = -np.linspace(10, 20, D.shape[0])

    def distance_2_RSS(P_i, D, alpha, d0):
        with np.errstate(divide='ignore'):
            s = 10 * alpha * np.log10((D) / d0)
        return P_i[:, np.newaxis] - s

    def RSS_2_distance(P_i, RSS, alpha, d0, sigma: float = 0.2, add_noise: bool = True):
        if add_noise:
            noise_matrix = np.random.lognormal(mean=0, sigma=sigma, size=RSS.shape)
            noise_matrix = (noise_matrix + noise_matrix.T) / 2  # Ensure symmetry
            RSS += noise_matrix
            
        with np.errstate(divide='ignore', invalid='ignore'):
            d = d0 * 10 ** ((P_i[:, np.newaxis] - RSS) / (10 * alpha))
        np.fill_diagonal(d, 0)
        return d

    RSS = distance_2_RSS(P_i, D, alpha, d0)
    DD = RSS_2_distance(P_i, RSS, alpha, d0, sigma=noise, add_noise=True)

    DD[:n_anchors, :n_anchors] = D[:n_anchors, :n_anchors]
    DD = np.abs(DD) * B

    """ J = calculate_jacobian(X_true, DD)
    FIM = calculate_fim(J, noise**2)
    CRLB = np.linalg.inv(FIM)

    # Extract variances (diagonal elements) and calculate overall CRLB
    #vv = np.array([CRLB[2*x, 2*x] + CRLB[2*x + 1, 2*x + 1] for x in range(X_true.shape[0])])
    variances = np.diag(CRLB)
    overall_crlb = np.sqrt(np.sum(variances))
    print("Overall CRLB:", overall_crlb) """

    #fim = derivative_n(RSS, DD, 3, P_i, noise**2)
    #print(fim)

    return DD, B

def derivative_n(rss_measurements, distances, n, P_i, sigma_squared):
    # Create a mask for distances greater than zero
    mask = distances > 0
    # Apply the mask to distances and rss_measurements
    masked_distances = distances[mask]
    masked_rss_measurements = rss_measurements[mask]
    # Calculate the derivative using only the masked values
    return np.sum(10 * np.log10(masked_distances) * (masked_rss_measurements - (P_i[:, np.newaxis] - 10 * n * np.log10(masked_distances))) / sigma_squared)

def calculate_jacobian(X_est, D):
    n = X_est.shape[0]  # Number of points
    J = np.zeros((n, n, 2, 2))  # Initialize the Jacobian matrix

    for i in range(n):
        for j in range(n):
            if i != j:
                xi, yi = X_est[i]
                xj, yj = X_est[j]
                d_ij = D[i, j]  # Use the estimated distance from RSS measurements
                
                # Avoid division by zero
                if d_ij == 0:
                    continue
                
                J[i, j, 0, 0] = (xi - xj) / d_ij
                J[i, j, 1, 1] = (yi - yj) / d_ij
                J[j, i, 0, 0] = -(xi - xj) / d_ij
                J[j, i, 1, 1] = -(yi - yj) / d_ij

    return J

def calculate_fim(J, sigma_squared):
    n = J.shape[0]  # Number of points
    FIM = np.zeros((2*n, 2*n))  # Initialize the FIM

    for i in range(n):
        for j in range(n):
            if i != j:
                J_ij = J[i, j]  # Jacobian submatrix for points i and j
                # Contribution to the FIM from points i and j
                FIM_contribution = np.dot(J_ij.T, J_ij) / sigma_squared
                # Add contribution to the total FIM
                # Assuming that coordinates are ordered as [x1, y1, x2, y2, ..., xn, yn]
                idx = slice(2*i, 2*i+2)  # Index slice for point i
                idy = slice(2*j, 2*j+2)  # Index slice for point j
                FIM[idx, idx] += FIM_contribution[:, :2]
                FIM[idy, idy] += FIM_contribution[:, :2]

    return FIM


def plot_RSS_over_distance(PL_d0: float = 40, P_i: float = 0, PLE: float = 3.0, d0: float = 1.0, distance_max: float = 100, num_points: int = 1000, shadowing: float = 2.0, name: str = "Bluetooth", color: str = "blue"):
    distances = np.linspace(d0, distance_max, num_points)
    epsilon = 1e-9
    
    def distance_2_RSS(P_i, D, alpha, d0, epsilon):
        s = PL_d0 + 10 * alpha * np.log10((D + epsilon) / d0)
        return P_i - s
    
    RSS_values = distance_2_RSS(P_i, distances, PLE, d0, epsilon)
    
    shadowing_noise = np.random.normal(0, shadowing, RSS_values.shape)
    RSS_values_with_shadowing = RSS_values + shadowing_noise*0
    
    # Define RSS thresholds for signal levels
    excellent_threshold = -50
    good_threshold = -60
    fair_threshold = -70
    poor_threshold = -80
    
    # Plot shaded regions for signal levels
    plt.fill_between(distances, RSS_values_with_shadowing, excellent_threshold, where=(RSS_values_with_shadowing >= excellent_threshold), facecolor=color, alpha=0.1)
    plt.fill_between(distances, RSS_values_with_shadowing, good_threshold, where=(RSS_values_with_shadowing < excellent_threshold) & (RSS_values_with_shadowing >= good_threshold), facecolor=color, alpha=0.3)
    plt.fill_between(distances, RSS_values_with_shadowing, fair_threshold, where=(RSS_values_with_shadowing < good_threshold) & (RSS_values_with_shadowing >= fair_threshold), facecolor=color, alpha=0.5)
    plt.fill_between(distances, RSS_values_with_shadowing, poor_threshold, where=(RSS_values_with_shadowing < fair_threshold) & (RSS_values_with_shadowing >= poor_threshold), facecolor=color, alpha=0.7)
    plt.fill_between(distances, RSS_values_with_shadowing, min(RSS_values_with_shadowing), where=(RSS_values_with_shadowing < poor_threshold), facecolor=color, alpha=0.9)
    
    plt.plot(distances, RSS_values_with_shadowing, label=name, color=color)
    return RSS_values_with_shadowing


def get_graphs(D: np.ndarray) -> dict:
    graphs = {}
    graphs["full"] = D
    one_hop = D.copy()
    graphs["one"] = one_hop

    G = nx.from_numpy_array(one_hop)
    two_hop = one_hop.copy()
    for j, paths in nx.all_pairs_shortest_path(G, 2):
        for q, _ in paths.items():
            two_hop[j, q] = nx.shortest_path_length(G, j, q, weight='weight')
    
    graphs["two"] = two_hop
    return graphs

def get_n_hop(X: np.ndarray, D: np.ndarray, n: int, r: int, n_anchors, nth_n):
    C = np.zeros_like(D)
    for i in range(1, n+1):
        n_hop_D = display_n_hop(D, X, i)
        n_hop_mask = (n_hop_D > 0) & (C == 0)
        C[n_hop_mask] = 1
        if i == nth_n:
            B = n_hop_D > 0
    return n_hop_D, B, C

def display_n_hop(network: np.ndarray, X: np.ndarray, n: int):
    n_hop = network.copy()
    G = nx.from_numpy_array(n_hop)
    if n == 1:
        return n_hop

    for j, paths in nx.all_pairs_shortest_path(G, n):
        for q, path in paths.items():
            n_hop[j, q] = nx.shortest_path_length(G, j, q, weight='weight')

    return n_hop

""" def get_n_hop(X: np.ndarray, D: np.ndarray, n: int, r: int, n_anchors, nth_n):
    #sns.set(style="whitegrid")
    #fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot anchors
    #ax.scatter(X[:n_anchors, 0], X[:n_anchors, 1], marker="*", c="r", label=r"$N_{a}$", s=150)
    #for i in range(n_anchors):
    #    ax.annotate(rf"$A_{{{i}}}$", (X[i, 0], X[i, 1]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=12, color='r')

    # Plot targets
    #ax.scatter(X[n_anchors:, 0], X[n_anchors:, 1], marker="+", c="g", label=r"$N_{t}$", s=150)
    #for i in range(n_anchors, len(X)):
    #    ax.annotate(rf"$T_{{{i - n_anchors}}}$", (X[i, 0], X[i, 1]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=12, color='g')
    
    colors = ["black", "blue", "yellow", "red", "green"]
    cumulative_edges = set()

    for i in range(1, n+1):
        new_edges, n_hop_D = display_n_hop(D, X, i, cumulative_edges, color=colors[i-1], ax=None)
        if i == nth_n:
            B = n_hop_D > 0
        cumulative_edges.update(new_edges)
    
    #ax.legend()
    #plt.tight_layout()
    #plt.title("N-hop distances")
    #plt.show()
    return n_hop_D, B

def display_n_hop(network: np.ndarray, X: np.ndarray, n: int, cumulative_edges, color: str, ax=None):
    n_hop = network.copy()
    G = nx.from_numpy_array(n_hop)
    if n == 1:
        return n_hop
    new_edges = set()
    #print("works")
    for j, paths in nx.all_pairs_shortest_path(G, n):
        for q, path in paths.items():
            if len(path) - 1 == n:  # Check if the path length is exactly n hops
                n_hop[j, q] = nx.shortest_path_length(G, j, q, weight='weight')

                edge = (j, q) if j < q else (q, j)
                if edge not in cumulative_edges:
                    new_edges.add(edge)
    #nG = nx.from_numpy_array(n_hop)
    #pos = {i: (X[i, 0], X[i, 1]) for i in range(len(X))}
    #nx.draw_networkx_edges(nG, pos=pos, edgelist=new_edges, width=2, edge_color=color, ax=ax, label=f"{n}-hop")
    
    return new_edges, n_hop """

def display_n_hop2(label_set, network: np.ndarray, X: np.ndarray, n: int, cumulative_edges, color: str, ax):
    n_hop = network.copy()
    G = nx.from_numpy_array(n_hop)
    new_edges = set()

    for j, paths in nx.all_pairs_shortest_path(G, cutoff=n):
        for q, path in paths.items():
            n_hop[j, q] = nx.shortest_path_length(G, j, q, weight='weight')
            edge = (j, q) if j < q else (q, j)
            if edge not in cumulative_edges:
                new_edges.add(edge)
    nG = nx.from_numpy_array(n_hop)
    pos = {i: (X[i, 0], X[i, 1]) for i in range(len(X))}

    label = f"{n}-hop"
    if n == 1 or n == 4:
        if label not in label_set or n==4:
            label_set.add(label)
            nx.draw_networkx_edges(nG, pos=pos, edgelist=new_edges, width=2, edge_color=color, ax=ax, label=label)
        else:
            nx.draw_networkx_edges(nG, pos=pos, edgelist=new_edges, width=2, edge_color=color, ax=ax)
            
    return new_edges, n_hop

def display():
    sns.set(style="whitegrid")
    G = nx.Graph()
    cumulative_edges = set()
    pos = {}
    label_set = set()
    fig, ax = plt.subplots(figsize=(10,6))
    ax.set_xlim(0, 135)
    ax.set_ylim(0, 135)
    r = 20
    n = 4
    colors = ["black", "blue", "yellow", "red", "green"]

    def onclick(event):
        if event.inaxes is not None:
            node_id = len(G.nodes)
            G.add_node(node_id, pos=(event.xdata, event.ydata))
            pos[node_id] = (event.xdata, event.ydata)

            #ax.clear()

            ax.grid(True)
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            
            nx.draw(G, pos, ax=ax, with_labels=True, node_color='skyblue', node_size=500, font_weight='bold')

            X = np.array([pos[i] for i in range(len(pos))])
            network = euclidean_distances(X)
            network[network > r] = 0

            label = "communication range of 0"
            if label not in label_set:
                label_set.add(label)
                circle = plt.Circle((X[0, 0], X[0, 1]), r, color='b', linestyle='--', linewidth=2, alpha=0.4, fill=False, label="communication range of 0")
                plt.gca().add_artist(circle)
            else:
                circle = plt.Circle((X[0, 0], X[0, 1]), r, color='b', linestyle='--', linewidth=2, alpha=0.4, fill=False)
                plt.gca().add_artist(circle)
            for i in range(n):
                new_edges, n_hop = display_n_hop2(label_set, network, X, n=i+1, cumulative_edges=cumulative_edges, color=colors[i], ax=ax)
                cumulative_edges.update(new_edges)

            if len(G) == 5:
                circle = plt.Circle((X[0, 0], X[0, 1]), n_hop[0, -1], color='r', linestyle='--', linewidth=2, alpha=0.4, fill=False, label="distance from 0 to 4 after n-hops")
                plt.gca().add_artist(circle)
                plt.legend()
            plt.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()


def relative_spread(particles_u: np.ndarray, particles_r: np.ndarray, d_ru: float):
    dist_ur = particles_u - particles_r
    angle_samples = np.arctan2(dist_ur[:, 1], dist_ur[:, 0])
    
    # Add 2π, 0, -2π for 2π-periodicity
    extended_samples = np.concatenate([angle_samples, angle_samples + 2*np.pi, angle_samples - 2*np.pi])
    kde = gaussian_kde(extended_samples)
    
    samples = kde.resample(particles_u.shape[0]).T
    samples = np.mod(samples + np.pi, 2*np.pi) - np.pi  # Ensure samples are in [-π, π]
    w_xy = kde(samples.T) #+ 1e-7  # Prevent division by zero
    
    particle_noise = np.random.normal(0, 1, size=particles_u.shape[0]) * 1
    cos_u = (d_ru + particle_noise).reshape(-1, 1) * np.cos(samples)
    sin_u = (d_ru + particle_noise).reshape(-1, 1) * np.sin(samples)
    d_xy = np.column_stack([cos_u, sin_u])
    
    return d_xy, w_xy

def random_spread(particles_r: np.ndarray, d_ru: float):
    particle_noise = np.random.normal(0, 1, size=particles_r.shape[0]) * 1
    thetas = np.random.uniform(0, 2*np.pi, size=particles_r.shape[0])
    cos_u = (d_ru + particle_noise) * np.cos(thetas)
    sin_u = (d_ru + particle_noise) * np.sin(thetas)
    d_xy = np.column_stack([cos_u, sin_u])
    w_xy = np.ones(shape=particles_r.shape[0])

    return d_xy, w_xy


def plot_techs():
    # Define the data
    technologies = ['Bluetooth', 'Wi-Fi', 'Cellular', 'UWB']
    categories = ['Excellent', 'Good', 'Fair', 'Poor', 'Unusable']
    rss_ranges = {
        'Bluetooth': [-50, -60, -70, -90, -91],
        'Wi-Fi': [-50, -60, -70, -80, -81],
        'Cellular': [-70, -85, -100, -110, -111],
        'UWB': [-50, -60, -70, -80, -81]
    }
    distances = {
        'Bluetooth': [10, 20, 30, 100, 101],
        'Wi-Fi': [15, 25, 50, 75, 76],
        'Cellular': [300, 1000, 5000, 10000, 10001],
        'UWB': [10, 20, 30, 50, 51]
    }

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set the bar width
    bar_width = 0.15

    # Set the positions of the bars
    r1 = np.arange(len(categories))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]

    # Plot the bars
    ax.bar(r1, distances['Bluetooth'], color='b', width=bar_width, edgecolor='grey', label='Bluetooth')
    ax.bar(r2, distances['Wi-Fi'], color='r', width=bar_width, edgecolor='grey', label='Wi-Fi')
    ax.bar(r3, distances['Cellular'], color='g', width=bar_width, edgecolor='grey', label='Cellular')
    ax.bar(r4, distances['UWB'], color='c', width=bar_width, edgecolor='grey', label='UWB')

    # Add labels
    ax.set_xlabel('RSS Quality', fontweight='bold')
    ax.set_ylabel('Effective Distance (m)', fontweight='bold')
    ax.set_title('Effective Distance for Different RSS Quality Levels', fontweight='bold')
    ax.set_xticks([r + 1.5 * bar_width for r in range(len(categories))])
    ax.set_xticklabels(categories)

    # Add a legend
    ax.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()

from scipy.stats import t

def correlation_p_value(r, n):
    """
    Calculate the p-value for a given Pearson correlation coefficient.
    
    Parameters:
    r (float): Pearson correlation coefficient.
    n (int): Sample size.
    
    Returns:
    float: p-value indicating the statistical significance of the correlation.
    """
    # Calculate the t-statistic
    t_stat = r * np.sqrt((n - 2) / (1 - r**2))
    
    # Degrees of freedom
    df = n - 2
    
    # Calculate the p-value (two-tailed test)
    p_value = 2 * (1 - t.cdf(np.abs(t_stat), df=df))
    
    return p_value


def difference_of_distances(preds: np.ndarray, trues: np.ndarray, radius: int, B: np.ndarray):
    # Compute distance matrices
    preds_D = euclidean_distances(preds) * B
    trues_D = euclidean_distances(trues) * B

    # Apply radius cutoff
    preds_D[preds_D > radius] = 0
    trues_D[trues_D > radius] = 0

    # Compute differences and normalize by true distances
    difference = np.abs(trues_D - preds_D)
    normalized_difference = np.divide(difference, trues_D, out=np.zeros_like(difference), where=trues_D!=0)
    normalized_difference *= B  # Apply the connectivity mask

    # Sum of differences for each node
    sum_over = np.sum(normalized_difference, axis=1)

    # Number of neighbors for each node
    neighbors = np.count_nonzero(trues_D, axis=1)

    # Filter out nodes with zero neighbors to avoid division by zero
    non_zero_neighbors_idx = neighbors > 0
    filtered_neighbors = neighbors[non_zero_neighbors_idx]
    filtered_sum_over = sum_over[non_zero_neighbors_idx]

    # Calculate average difference per neighbor
    avg_difference_per_neighbor = filtered_sum_over / filtered_neighbors


    # Plotting the results
    #plt.figure(figsize=(10, 6))

    

    z = np.polyfit(filtered_neighbors, avg_difference_per_neighbor, 1)
    p = np.poly1d(z)
    x = np.linspace(min(filtered_neighbors), max(filtered_neighbors))
    correlation = np.corrcoef(filtered_neighbors, avg_difference_per_neighbor)[0, 1]
    p_val = correlation_p_value(correlation, preds.shape[0])
    """ plt.plot(x, p(x), color="red", label=f"Correlation: {correlation:.2f}, p_val: {p_val:.3}")

    scatter = plt.scatter(filtered_neighbors, avg_difference_per_neighbor, c=filtered_neighbors, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label="Number of Neighbors")
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Average Normalized Distance Difference")
    plt.title("Impact of Number of Neighbors on Relative Positioning Accuracy")
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show() """

    # Correlation Analysis
    
    #print(f"Correlation between number of neighbors and average normalized distance difference: {correlation:.2f}")



    #mean_of = np.mean(sum_over)

    #print(f"neighboring_distance difference: {sum_over}")
    #print(f"mean difference: {mean_of}")
    return avg_difference_per_neighbor

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
    
def weighted_covariance(particles: np.ndarray, weights: np.ndarray, nodes: np.ndarray, uncertainties_dict, overall_uncertainties_list, iteration):
    uncertainties = []
    for t, node in enumerate(nodes):
        diff = particles[t] - node
        weighted_diff = diff.T * weights[t]
        covariance_matrix = np.dot(weighted_diff, diff) / np.sum(weights[t])
        uncertainty = np.sqrt(np.trace(covariance_matrix))  # Single uncertainty value for the node
        uncertainties.append(uncertainty)
        uncertainties_dict[t].append(uncertainty)
        #print(f"Uncertainty {t} of {node}: {uncertainty}")
    
    overall_uncertainty = np.mean(uncertainties)  # Overall uncertainty as the mean of individual uncertainties
    overall_uncertainties_list.append(overall_uncertainty)
    #print(f"Overall Uncertainty: {overall_uncertainty}")
    return uncertainties, overall_uncertainty

def weighted_covariance_2(particles: np.ndarray,
                          weights: np.ndarray,
                          nodes: np.ndarray,
                          uncertainties_dict: dict,
                          overall_uncertainties_dict: dict,
                          iteration: int,
                          indicex: dict,
                          X_true: np.ndarray,
                          error_dict: dict) -> dict:
    uncertainties = []

    # Calculate uncertainties for each node
    for t, node in enumerate(nodes):
        diff = particles[t] - node
        weighted_diff = diff.T * weights[t]
        covariance_matrix = np.dot(weighted_diff, diff) / np.sum(weights[t])
        uncertainty = np.sqrt(np.trace(covariance_matrix))  # Single uncertainty value for the node
        uncertainties.append(uncertainty)
        uncertainties_dict[t].append(uncertainty)
    
    # Calculate average uncertainties for each key in indicex
    error = ERR(X_true, nodes)
    for key, indices in indicex.items():
        uncertainties_key = [uncertainties[idx] for idx in indices]
        error_key = [error[idx] for idx in indices]
        if uncertainties_key:
            average_uncertainty = np.mean(uncertainties_key)
            average_error = np.mean(error_key)
        else:
            average_uncertainty = 0  # Handle cases where there are no valid indices
            average_error = 0  # Handle cases where there are no valid indices
        
        if key not in overall_uncertainties_dict:
            overall_uncertainties_dict[key] = []
            error_dict[key] = []
        
        overall_uncertainties_dict[key].append(average_uncertainty)
        error_dict[key].append(average_error)
    
    return overall_uncertainties_dict, error_dict

def plot_uncertainties2(overall_uncertainties_dict: dict, label: str):
    plt.figure(figsize=(10, 6))

    for key, uncertainties in overall_uncertainties_dict.items():
        plt.plot(uncertainties, label=f'{key} anchor neighbors')
    
    plt.xlabel('Iteration')
    plt.ylabel(f'Average {label}')
    plt.title(f'Average {label} Over Iterations')

    plt.legend()
    plt.grid(True)
    plt.show()


def plot_uncertainties(uncertainties_dict, overall_uncertainties_list, n_anchors, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    # Plot each node's uncertainties
    for node, uncertainties in uncertainties_dict.items():
        #if node == 102 - n_anchors:
        #    ax.plot(uncertainties, label="highest")
        #else:
        ax.plot(uncertainties)

    # Plot overall uncertainty
    ax.plot(overall_uncertainties_list, label='Overall Uncertainty', linewidth=2, color='black')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Uncertainty')
    ax.set_title('Uncertainties Over Iterations')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    return ax

""" def weighted_covariance(particles: np.ndarray, weights: np.ndarray, nodes: np.ndarray, confidence_level=0.95):
    uncertainties = []
    chi2_val = chi2.ppf(confidence_level, df=2)  # df=2 for 2D space

    for t, node in enumerate(nodes):
        diff = particles[t] - node
        weighted_diff = diff.T * weights[t]
        covariance_matrix = np.dot(weighted_diff, diff) / np.sum(weights[t])
        uncertainty_radius = np.sqrt(chi2_val * np.trace(covariance_matrix))
        uncertainties.append(uncertainty_radius)
        print(f"Estimate {t}: {node}")
        print(f"Weighted Covariance Matrix:\n{covariance_matrix}")
        print(f"Uncertainty Radius {t} (95% Confidence): {uncertainty_radius}")
    
    overall_uncertainty = np.mean(uncertainties)  # Overall uncertainty as the mean of individual uncertainties
    print(f"Overall Uncertainty (95% Confidence): {overall_uncertainty}")
    return uncertainties, overall_uncertainty """



def plot_networks(X_true:np.ndarray, n_anchors: int, graphs: dict):
    fig, axs = plt.subplots(len(graphs), 1,  sharex=True, sharey=True, figsize=(6, 18))
    for (title, graph), ax in zip(graphs.items(), axs):
        ax.scatter(X_true[:n_anchors, 0], X_true[:n_anchors, 1], marker="*", c="r", label="anchors")
        ax.scatter(X_true[n_anchors:, 0], X_true[n_anchors:, 1], marker="+", c="g", label="true")
        nx.draw_networkx_edges(nx.from_numpy_array(graph), pos=X_true, ax=ax, width=0.5)
        ax.legend()
        ax.set_title(title)
    plt.show()

def plot_network(X_true: np.ndarray, B: np.ndarray,
                  n_anchors: int, r: float = None,
                    alpha: float = 0.6, subset=None,
                      D: np.ndarray = None, zoom: int = None,
                        ax = None, name:str = None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    
    G = nx.from_numpy_array(D)
    pos = {i: (X_true[i][0], X_true[i][1]) for i in range(len(X_true))}

    
    two_hop = D.copy()
    two_hop[two_hop < r] = 0
    # Draw 2-hop neighbor edges (beyond radius r)
    #nx.draw_networkx_edges(nx.from_numpy_array(two_hop), pos, edge_color='blue', width=1.337, alpha=alpha, ax=ax)

    # Draw immediate neighbor edges (within radius r)
    one_hop = D.copy()
    one_hop *= B
    one_hop[one_hop > r] = 0
    nx.draw_networkx_edges(nx.from_numpy_array(one_hop), pos, edge_color='black', width=1.337, alpha=alpha, ax=ax)
    
    # Plot anchors
    ax.scatter(X_true[:n_anchors, 0], X_true[:n_anchors, 1], marker="*", c="r", label=r"$N_{a}$", s=150)
    for i in range(n_anchors):
        ax.annotate(rf"$A_{{{i}}}$", (X_true[i, 0], X_true[i, 1]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=12, color='r')

    ax.legend(fontsize=14, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.7)
    #plt.xlabel("X-coordinate", fontsize=14)
    #plt.ylabel("Y-coordinate", fontsize=14)
    #plt.title("1-Hop True Connectivity Network", fontsize=16)
    #plt.title("True Graph", fontsize=16)
    ax.set_title(name, fontsize=16)
    plt.show()




def compare_networks(X_true, estimates, true_graphs, estimated_graphs, n_anchors):
    fig, axs = plt.subplots(len(true_graphs), 2, sharex=True, sharey=True, figsize=(12, 18))
    
    for i, (title, true_graph) in enumerate(true_graphs.items()):
        # Plot true graph
        ax = axs[i, 0]
        ax.scatter(X_true[:n_anchors, 0], X_true[:n_anchors, 1], marker="*", c="r")
        nx.draw_networkx_edges(nx.from_numpy_array(true_graph), pos=X_true, ax=ax, width=0.5, label="abc")
        ax.set_title(f"True: {title}")
        
        # Plot estimated graph
        ax = axs[i, 1]
        estimated_graph = estimated_graphs[title]
        ax.scatter(estimates[:n_anchors, 0], estimates[:n_anchors, 1], marker="*", c="r")
        nx.draw_networkx_edges(nx.from_numpy_array(estimated_graph), pos=estimates, ax=ax, width=0.5)
        ax.set_title(f"Estimated: {title}")
    
    plt.show()


def plot_results_initial(anchors, X, weighted_means, intersections, M):
    n_anchors = anchors.shape[0]
    plt.scatter(anchors[:, 0], anchors[:, 1], label="anchors", c="r", marker='*') # anchors nodes
    plt.scatter(X[n_anchors:, 0], X[n_anchors:, 1], label="true", c="g", marker='P') # target nodes
    plt.scatter(weighted_means[:, 0], weighted_means[:, 1], label="preds", c="y", marker="X") # preds
    plt.plot([X[n_anchors:, 0], weighted_means[:, 0]], [X[n_anchors:, 1], weighted_means[:, 1]], "k--")
    for i, xt in enumerate(X):
        if i < n_anchors:
            plt.annotate(f"A_{i}", xt)
        else:
            """ if not (i > 33 and i < 35):
                continue """
            
            """ bbox = intersections[i]
            xmin, xmax, ymin, ymax = bbox
            plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin])
            plt.annotate(f"t_{i}", xt) """
            """ plt.scatter(M[i, :, 0], M[i, :, 1], marker=".", s=10)
            plt.annotate(f"t_{i}", xt)
            plt.annotate(f"p_{i}", weighted_means[i - n_anchors]) """
    plt.legend()
    plt.title(f"iter: {iter}, Predictions with initial weights")
    plt.show()

def plot_probabilistic_vs_deterministic(X: np.ndarray, anchors: np.ndarray, weights: np.ndarray, weighted_means: np.ndarray, all_particles: np.ndarray):

    test_node = 5
    #additional = np.array([[-15, 2], [20, 2]])
    #weighted_means += additional
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True) # create a figure and two subplots
    n_anchors = anchors.shape[0]
                
    pd_xru = pandas.DataFrame(all_particles[test_node], columns=["X", "Y"])
    axes[0].scatter(anchors[:, 0], anchors[:, 1], label="anchors", c="r", marker='*') # plot the anchors on the first subplot
    sns.kdeplot(data=pd_xru, x="X", y="Y", fill=True, common_norm=True, weights=weights[test_node], bw_method='silverman', gridsize=50, ax=axes[0]) # plot the kernel density estimate on the first subplot
    axes[0].scatter(X[n_anchors:, 0], X[n_anchors:, 1], label="true", c="g", marker='P') # plot the true positions on the first subplot
    axes[0].scatter(weighted_means[:, 0], weighted_means[:, 1], label="predictions", c="y", marker="X") # plot the predictions on the first subplot
    axes[0].plot([X[n_anchors:, 0], weighted_means[:, 0]], [X[n_anchors:, 1], weighted_means[:, 1]], "k--",) # plot the error lines on the first subplot
    nx.draw_networkx_edges(nx.from_numpy_array(D), pos=X_true, width=0.5, ax=axes[0]) # plot the network edges on the first subplot
    axes[0].set_title("Probabilistic") # set the title for the first subplot
    axes[0].set_xlabel(None) # remove the x-axis label for the first subplot
    axes[0].set_ylabel(None) # remove the y-axis label for the first subplot

    mds_X = X.copy()
    mds = ClassicMDS(X=mds_X, m_anchors=n_anchors, noise=1)
    mds_pred = mds.classic_mds()
    mds_ab = mds.least_squares_registration(anchors=mds_X[:n_anchors].copy(), anchors_hat=mds_pred[:mds.m_anchors].copy(), X_hat=mds_pred)
    axes[1].scatter(mds_X[:n_anchors, 0], mds_X[:n_anchors, 1], label="anchors", c="r", marker='*') # plot the anchors on the second subplot
    axes[1].scatter(mds_X[n_anchors:, 0], mds_X[n_anchors:, 1], label="true", c="g", marker='P') # plot the true positions on the second subplot
    axes[1].scatter(mds_ab[n_anchors:, 0], mds_ab[n_anchors:, 1], label="predictions", c="y", marker="X") # plot the predictions on the second subplot
    axes[1].plot([mds_X[n_anchors:, 0], mds_ab[n_anchors:, 0]], [mds_X[n_anchors:, 1], mds_ab[n_anchors:, 1]], "k--",) # plot the error lines on the second subplot
    nx.draw_networkx_edges(nx.from_numpy_array(euclidean_distances(X)), pos=mds_X, width=0.5, ax=axes[1]) # plot the network edges on the second subplot
    axes[1].set_title("Deterministic") # set the title for the second subplot
    axes[1].set_xlabel(None) # remove the x-axis label for the second subplot
    axes[1].set_ylabel(None) # remove the y-axis label for the second subplot

    for j, p in enumerate(weighted_means):
        plt.annotate(f"{j+n_anchors}", p)
    plt.show()


def plot_proposal_dist(all_particles: np.ndarray, weights: np.ndarray, X: np.ndarray, D):
    fig, ax = plt.subplots()
    for i, Xr in enumerate(all_particles):
        if i < 4:
            continue
        pd_xru = pandas.DataFrame(Xr, columns=["x", "y"])
        sns.kdeplot(data=pd_xru, x="x", y="y", common_norm=False,
                    fill=True,
                    weights=weights[i],
                    bw_method='silverman',
                    gridsize=50,
                    cbar=False,
                    alpha=0.3,
                    ax=ax)
    """ plt.show()

    print(f"proposal: {all_particles[5]}")
    
    gg1 = np.array([-18, 10])
    test_node = 5
    pd_xru = pandas.DataFrame(all_particles[test_node] + gg1, columns=["x", "y"])
    sns.kdeplot(data=pd_xru, x="x", y="y", common_norm=False,
                fill=True,
                weights=weights[test_node],
                bw_method='silverman',
                gridsize=50,
                cbar=False,
                color="red",
                alpha=0.3,
                ax=ax)
    gg2 = np.array([-18,-18])
    test_node_2 = 4
    pd_xru = pandas.DataFrame(all_particles[test_node_2] +gg2, columns=["x", "y"])
    sns.kdeplot(data=pd_xru, x="x", y="y", common_norm=False,
                fill=True,
                weights=weights[test_node_2],
                bw_method='silverman',
                gridsize=50,
                cbar=False,
                color="blue",
                alpha=0.3,
                ax=ax)

    gg3 = np.array([0, 4])
    test_node_2 = 6
    pd_xru = pandas.DataFrame(all_particles[test_node_2] + gg3, columns=["x", "y"])
    sns.kdeplot(data=pd_xru, x="x", y="y", common_norm=False,
                fill=True,
                weights=weights[test_node_2],
                bw_method='silverman',
                gridsize=50,
                cbar=False,
                color="yellow",
                alpha=0.3,
                ax=ax)
    gg4 = np.array([-18, 0])
    test_node_2 = 7
    pd_xru = pandas.DataFrame(all_particles[test_node_2] + gg4, columns=["x", "y"])
    sns.kdeplot(data=pd_xru, x="x", y="y", common_norm=False,
                fill=True,
                weights=weights[test_node_2],
                bw_method='silverman',
                gridsize=50,
                cbar=False,
                color="green",
                alpha=0.3,
                ax=ax)
    ppp = 50
    aa = 0.2
    ax.scatter(all_particles[test_node, ppp:, 0]+gg1[0], all_particles[test_node, ppp:, 1]+gg1[1], label="particles of 1", c="red", alpha=aa)
    #ax.scatter(m_ru[(4, 5)].dataset.T[ppp:, 0]+gg2[0], m_ru[(1, test_node)].dataset.T[ppp:, 1]+gg2[1], label="particles of 1", c="red", alpha=aa)
    ax.scatter(all_particles[test_node, ppp:, 0]+gg2[0]+10, all_particles[test_node, ppp:, 1]+gg2[1]+7, label="particles of 0", c="blue", alpha=aa)
    ax.scatter(all_particles[test_node, ppp:, 0]+gg3[0]+21, all_particles[test_node, ppp:, 1]+gg3[1]-3, label="particles of 6", c="yellow", alpha=aa)
    ax.scatter(all_particles[test_node, ppp:, 0]+gg4[0]+30, all_particles[test_node, ppp:, 1]+gg4[1]+10, label="particles of 4", c="green", alpha=aa)
    plt.scatter(X[:, 0], X[:, 1], label="nodes", c="gray", marker='o') # plot the true positions on the first subplot
    plt.scatter(X[test_node, 0], X[test_node, 1], label="x", c="orange", marker='o') """
    nx.draw_networkx_edges(nx.from_numpy_array(D), pos=X, width=0.5, ax=ax)

    for j, p in enumerate(X):
        plt.annotate(f"{j}", p)
    # Create a patch for the legend
    blue_patch = mpatches.Patch(color='blue', label='PDF estimate of RV x')
    #handles, labels = ax.get_legend_handles_labels()
    #handles.append(blue_patch)
    #ax.legend(handles=handles)
    ax.legend()
    plt.show()


def plot_sellected_particles_of(intersections: np.ndarray, Mu_new: np.ndarray, all_particles: np.ndarray, node: int):
    bbox = intersections[node]
    xmin, xmax, ymin, ymax = bbox
    plt.scatter(Mu_new[:, 0], Mu_new[:, 1], s=9, label=f"all particles of {node}")
    plt.scatter(all_particles[node, :, 0], all_particles[node, :, 1], s=9, label=f"particles of {node}")
    plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin])
    plt.title(f"Sellected particles of {node}")
    plt.legend()
    plt.show()

def plot_RMSE(_rmse, iter, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    top = max(rmse[0] for rmse in _rmse)  # Get the maximum RMSE value
    #ax.ylim((0, top))  # Set the y-axis limits
    ax.set_ylim((0, top))
    ax.plot(np.arange(iter), [rmse[0] for rmse in _rmse], color="blue", label="RMSE")  # Plot only the RMSE values
    ax.plot(np.arange(iter), [rmse[1] for rmse in _rmse], color="blue", linestyle="--", label="Median")  # Median
    ax.plot(np.arange(iter), [rmse[2] for rmse in _rmse], color="red", label="CRLB")  # Plot benchmark
    ax.set_ylabel("RMSE")
    ax.set_xlabel("iteration")
    #plt.tight_layout()
    ax.legend()
    #plt.savefig(f"{folder}/rmseplot.png")
    #plt.show()
    return ax
    

def plot_sampling_from_kde_ur(anchor_list, u, r, X, x_ru, Mu, Mr, sampled_particles, w_ru):
    if u not in anchor_list and r not in anchor_list:
        plt.scatter(X[u, 0], X[u, 1], label="node u", c="r", marker='*')
        plt.scatter(X[r, 0], X[r, 1], label="node r", c="g", marker="+")
        plt.annotate(f"receiver {u}", X[u])
        plt.annotate(f"sender {r}", X[r])

        plt.scatter(x_ru[:, 0], x_ru[:, 1], marker=".", s=45, c="b", label="kde")
        plt.scatter(Mu[:, 0], Mu[:, 1], marker=".", s=45, c="pink", label=f"particles of {u}")
        plt.scatter(Mr[:, 0], Mr[:, 1], marker=".", s=45, c="orange", label=f"particles of {r}")

        plt.scatter(sampled_particles[:, 0], sampled_particles[:, 1], marker=".", s=55, c="c", label="sampels")
        for j, p in enumerate(x_ru):
            plt.annotate("{:.2f}".format(w_ru[j]), p)
        # We still need x_ru for anchors because they send messages
        
        plt.legend()
        plt.show()

def plot_kde_ru(kde, Mu, intersections, u, r, x_ru, w_ru):
    plt.scatter(kde.dataset.T[:, 0], kde.dataset.T[:, 1], label=f"kde of {r}")
    plt.scatter(Mu[:, 0], Mu[:, 1], label=f"particles of {u}")

    bbox = intersections[u]
    xmin, xmax, ymin, ymax = bbox
    plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin])
    plt.title(f"Sellected particles of {u}")

    plt.legend()
    plt.show()

    pd_xru = pandas.DataFrame(x_ru, columns=["X", "Y"])
    # Create the KDE plot
    sns.kdeplot(data=pd_xru, x="X", y="Y", fill=True, common_norm=True, weights=w_ru)
    plt.show()
    # kde constructed with particles of r to evaluate resampled particles of u

def plot_exception(W_u, u, intersections, X, Mu_new, indices):
    print(W_u)
    print(f"iteration: {iter}, u: {u}")
    bbox = intersections[u]
    xmin, xmax, ymin, ymax = bbox
    plt.scatter(X[u, 0], X[u, 1], s=15, c="green", label=f"true {u} pos")
    plt.scatter(Mu_new[:, 0], Mu_new[:, 1], s=20, c="blue", label=f"particles of {u}")
    plt.scatter(Mu_new[indices, 0], Mu_new[indices, 1], s=21, c="red", label=f"choosen particles of M{u}_new")
    plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin])
    plt.title(f"Particles of {u}")
    plt.legend()
    plt.show()

# Assuming X, a, D, and X_true are defined elsewhere in your code



def plot_gaussian_ring(X, a, r, u, D, c):
    
    # Calculate the distance between the central node and the reference node
    distance = D[r, u]
    
    # Generate theta values
    thetas = np.linspace(0, 2*np.pi - np.pi/15, 30)
    
    # Calculate the coordinates of the points around the central node
    cos_u = distance * np.cos(thetas)
    sin_u = distance * np.sin(thetas)
    x_ru = X[r] + np.column_stack([cos_u, sin_u])
    
    # Create a DataFrame with the coordinates
    pd_xru = pandas.DataFrame(x_ru, columns=["x", "y"])
    
    # Plot the KDE
    sns.kdeplot(data=pd_xru, x="x", y="y", common_norm=False,
                fill=True,
                weights=np.ones_like(thetas),
                bw_method='silverman',
                bw_adjust=0.5,
                color=c,
                alpha=0.5,
                label=f"u"
                )
    
def plot_rings(X, anchors, D, intersections, radius, node):
    # Define the central node and the reference node
    # Plot the nodes
    n_anchors = anchors.shape[0]
    plt.figure(figsize=(10, 6))
    plt.xlim((5, 82))
    plt.ylim((25, 75))
    
    # Draw the edges
    #nx.draw_networkx_edges(nx.from_numpy_array(D), pos=X, width=1)
    G = nx.from_numpy_array(D)
    neighbors = [(n, node) for n in G.neighbors(node) if D[node, n] < radius]
    nx.draw_networkx_edges(G, pos=dict(zip(range(len(X)), X)), edgelist=neighbors, width=1.25)
    
    colors = ['b', 'g', 'y', 'c', 'o']
    counter = 0
    for i, nn in enumerate(G.neighbors(node)):
        if D[node, nn] < radius:
            plot_gaussian_ring(X, n_anchors, nn, node, D, colors[counter])
            counter += 1
    plt.legend()
    plt.title(f"Likelihoods for states of $T_{{{node - n_anchors}}}$")
    plot_priors(X, D, node - n_anchors, anchors, intersections)
    plt.show()

def plot_priors(X: np.ndarray, D: np.ndarray, node, anchors: np.ndarray, intersections: np.ndarray, bboxes: np.ndarray=None, color=None):
    n_anchors = anchors.shape[0]
    node += n_anchors
    sns.set(style="whitegrid")
    #plt.figure(figsize=(6, 6))
    
    # Plot anchors
    plt.scatter(X[:n_anchors, 0], X[:n_anchors, 1], marker="*", c="r", label=r"$N_{a}$", s=150)
    for i in range(n_anchors):
        plt.annotate(rf"$A_{{{i}}}$", (X[i, 0], X[i, 1]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=12, color='r')

    # Plot targets
    plt.scatter(X[n_anchors:, 0], X[n_anchors:, 1], marker="+", c="g", label=r"$N_{t}$", s=150)
    for i in range(n_anchors, len(X)):
        plt.annotate(rf"$T_{{{i - n_anchors}}}$", (X[i, 0], X[i, 1]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=12, color='g')

    G = nx.from_numpy_array(D)
    edges = [(n, node) for n in G.neighbors(node) if n < n_anchors]
    if bboxes is not None:
        nx.draw_networkx_edges(G, pos=X, edgelist=edges, width=1)
        for neighbor, node in edges:
            xmin, xmax, ymin, ymax = bboxes[node][neighbor]
            plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], label=rf"bbox of $A_{{{neighbor}}}$")
    
    if color is None:
        color = "blue"

    colors = ["red", "blue", "green", "yellow"]
    if intersections is not None:
        int_bbox = intersections[node]
        xmin, xmax, ymin, ymax = int_bbox
        plt.fill_between([xmin, xmax], ymin, ymax, color=colors[0], alpha=0.3, label=rf"Prior of $T_{{{node - n_anchors}}}$")


    """ for i in range(n_anchors, len(X)):
        if intersections is not None:
            int_bbox = intersections[i]
            xmin, xmax, ymin, ymax = int_bbox
            plt.fill_between([xmin, xmax], ymin, ymax, color=colors[i - n_anchors], alpha=0.3, label=rf"Prior of $T_{{{i - n_anchors}}}$")
             """
    """ if bboxes is not None:
        plt.title(rf"Prior of target $T_{{{node - n_anchors}}}$, 2-hop")
    else:
        plt.title("Intersecitons of anchor priors for each agent, 2-hop") """
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_detection_model(X, D, r, anchors, radius, intersections):
    # Define the bounds of the box for the uniform distribution
    x_min, x_max, y_min, y_max = 0, 85, 5, 80
    n_anchors = anchors.shape[0]
    
    # Create a uniform grid of points within the bounded box
    x = np.linspace(x_min, x_max, 400)
    y = np.linspace(y_min, y_max, 400)
    XX, YY = np.meshgrid(x, y)
    grid_points = np.vstack([XX.ravel(), YY.ravel()]).T

    # Create a figure and its axes
    fig, ax = plt.subplots(figsize=(9, 4))
    
    # Plot the network and the nodes
    G = nx.from_numpy_array(D)

    neighbors = [(n, r) for n in G.neighbors(r) if D[r, n] < radius]
    for uu, rr in neighbors:
        if uu != 5:
            continue
        probabilities = detection_probability(grid_points, X[uu], radius, D[rr, uu])
        Z = probabilities.reshape(XX.shape)
        cp = ax.contourf(XX, YY, Z, alpha=0.3, cmap='viridis', levels=25)

    nx.draw_networkx_edges(G, pos=dict(zip(range(len(X)), X)), edgelist=neighbors, width=0.5, ax=ax)
    
    nonneighbors = [(n, r) for n in G.neighbors(r) if D[r, n] > radius]
    for uu, rr in nonneighbors:
        probabilities = 1 - detection_probability(grid_points, X[uu], radius, None)
        Z = probabilities.reshape(XX.shape)
        cp = ax.contourf(XX, YY, Z, 50, alpha=0.5, cmap='viridis')
    nx.draw_networkx_edges(G, pos=dict(zip(range(len(X)), X)), edgelist=nonneighbors, width=0.5, ax=ax, edge_color='red')

    sm = ScalarMappable(cmap='viridis', norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Connectivity Probability")
    """ for i, x in enumerate(X):
        ax.annotate(f"{i}", x) """
    
    # Draw a circle around x_r with the specified radius
    """ circle = Circle(X[5], radius, color='blue', fill=False)
    ax.add_patch(circle) """

    # Set the title and labels
    #ax.set_title('Probability of connectivity Function with 2-hop Uniform Prior')
    ax.set_title(rf'Probability of connectivity over states of $T_{{{r - n_anchors}}}$ from $T_{{{6 - n_anchors}}}$')
    ax.grid(True)
    """ plt.xlim(0, 100)
    plt.ylim(0, 100) """
    # Show the plot
    #plot_priors(X, D, r - n_anchors, anchors, intersections)
    #plot_priors(X, D, 5, anchors, intersections, color='green')
    #plt.show()
    return ax

def plot_initial_particles(X, D, M, anchors, intersections, ax=None):
    n_anchors = anchors.shape[0]
    colors = ['blue', 'red', 'orange', 'purple', 'cyan', 'green', 'pink', 'yellow']
    for i in range(n_anchors, X.shape[0]):
        #if i != 5 and i != 4:
        #    continue
        #plot_priors(X, D, i, anchors, intersections, color=colors[i])
        plt.annotate(f"{i}", X[i])
        if ax is not None:
            ax.scatter(M[i, :, 0], M[i, :, 1], marker='.', c=colors[i], label=r"$X^{n}_{" + str(i) +"}$")
        else:
            plt.scatter(M[i, :, 0], M[i, :, 1], marker='.', c=colors[i], label=r"$X^{n}_{" + str(i) +"}$")
    
    #plt.title("Drawing particles from priors")
    #plt.show()

def plot_message_approx(X, D, M, anchors, intersections, x_ru, w_ru, r, u, radius, u_mean):
    if r == 4:
        for i, xr in enumerate(X):
            plt.annotate(f'{i}', xr)
        n_anchors = anchors.shape[0]
        X2 = X.copy()
        X2[u] = u_mean
        #plot_detection_model(X2, D, r, anchors, radius, intersections)
        plt.scatter(x_ru[:, 0], x_ru[:, 1], marker='.', c='blue', label=r"$X^{n+1}_{"+str(r)+r"\rightarrow "+str(u)+"}$")
        plot_initial_particles(X[:-2], D, M, anchors, intersections)
        pd_xru = pandas.DataFrame(x_ru, columns=["x", "y"])
        sns.kdeplot(data=pd_xru, x="x", y="y", common_norm=False,
            fill=True,
            weights=np.ones_like(w_ru),
            bw_method='silverman',
            color='cyan',
            bw_adjust=0.5,
            alpha=0.5,
            label="u"
            )
        plt.scatter(u_mean[0], u_mean[1], c='red', label=r"$\bar{x}_{" + str(u) + "}$")
        """ circle = Circle(X[r], D[r, u], fill=False)
        plt.gca().add_patch(circle) """

        plt.plot([X[r][0], X[u][0]], [X[r][1], X[u][1]], color='black', linestyle='-', linewidth=2)
        # Calculate the midpoint of the line for the annotation
        mid_x = (X[r][0] + X[u][0]) / 2
        mid_y = (X[r][1] + X[u][1]) / 2
        # Annotate the line with "d_{ij}"
        plt.annotate(r'$d_{ij}$', xy=(mid_x-2, mid_y-5), xytext=(mid_x-5, mid_y-8), textcoords='offset points', xycoords='data', ha='center', va='center')

        #plt.scatter(X[n_anchors:, 0], X[n_anchors:, 1], marker="o", c="gray", label="Nt")
        #plt.scatter(X[:n_anchors, 0], X[:n_anchors, 1], marker="*", c="red", label="Na")


        plt.title(r"Gaussian mixture estimate of message $m^{n+1}_{4 \rightarrow 5}$")
        handles, labels = plt.gca().get_legend_handles_labels()
        kde_handle = mpatches.Patch(color='cyan', label=r"$m^{n+1}_{i \rightarrow j}$")
        handles.append(kde_handle)
        plt.legend(handles=handles, labels=labels)
        plt.tight_layout()
        plt.show()

def plot_message_kde2(X, M, D, m_ru, anchors, u, radius, M_new, weights):
    n_anchors = anchors.shape[0]
    n_nodes = D.shape[0]
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))  # Adjust the figsize as needed
    
    G = nx.from_numpy_array(D)
    colors = ['blue', 'red', 'orange', 'purple', 'cyan', 'green', 'pink', 'yellow', "gray"]
    neighbors = list(G.neighbors(u))

    filtering = 50

    rr_kde4 = m_ru[(4, u)]
    x_r4 = rr_kde4.dataset.T
    x_r4 = x_r4[filtering:]

    rr_kde6 = m_ru[(6, u)]
    x_r6 = rr_kde6.dataset.T
    x_r6 = x_r6[filtering:]

    diff = np.linalg.norm(x_r6 - x_r4, axis=1)**2
    legend_handles = []
    print(len(M_new[u]))
    print(list(neighbors))
    # Loop over the neighbors
    for r in neighbors:
        if r != 4 and r != 6:
            continue
        r_kde = m_ru[(r, u)]
        Xr = r_kde.dataset.T
        Wr = r_kde.weights
        
        # Normalize weights to use as alpha values
        weights_t = weights[r, filtering:].copy()
        weights_t = np.exp(-diff / (radius**2))
        
        # Normalize weights_t to [0, 1] range
        weights_t -= np.min(weights_t)
        if np.max(weights_t) != 0:
            weights_t /= np.max(weights_t)
        
        # Plot particles with alpha values based on normalized weights
        ax.scatter(M_new[u][r%4][:, 0], M_new[u][r%4][:, 1], marker=".", label=rf"$X_{{{u}}}^{{n,{r} \rightarrow {u}}}$")
    
        pd_xru = pd.DataFrame(Xr, columns=["x", "y"])
        sns.kdeplot(data=pd_xru, x="x", y="y", common_norm=False,
                    fill=True,
                    weights=Wr,
                    bw_method='silverman',
                    bw_adjust=0.5,
                    color=colors[r % len(colors)],
                    alpha=0.2,
                    label=f"${{r}}$",
                    ax=ax)
        
    ax.set_title("Drawing particles from mixtures of messages")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_message_kde(X, M, D, m_ru, anchors, u, radius, M_new, weights):
    n_anchors = anchors.shape[0]
    n_nodes = D.shape[0]
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))  # Adjust the figsize as needed
    
    G = nx.from_numpy_array(D)
    colors = ['blue', 'red', 'orange', 'purple', 'cyan', 'green', 'pink', 'yellow', "gray"]
    neighbors = list(G.neighbors(u))

    filtering = 25

    rr_kde4 = m_ru[(4, u)]
    x_r4 = rr_kde4.dataset.T
    x_r4 = x_r4[filtering:]

    rr_kde6 = m_ru[(6, u)]
    x_r6 = rr_kde6.dataset.T
    x_r6 = x_r6[filtering:]

    diff = np.linalg.norm(x_r6 - x_r4, axis=1)**2
    legend_handles = []

    # Normalize weights to use as alpha values
    weights_t4 = weights[4, filtering:].copy()
    weights_t4 = np.exp(-diff / (radius**2))
    
    # Normalize weights_t to [0, 1] range
    weights_t4 -= np.min(weights_t4)
    if np.max(weights_t4) != 0:
        weights_t4 /= np.max(weights_t4)

    # Normalize weights to use as alpha values
    weights_t6 = weights[6, filtering:].copy()
    weights_t6 = np.exp(-diff / (radius**2))
    
    # Normalize weights_t to [0, 1] range
    weights_t6 -= np.min(weights_t6)
    if np.max(weights_t6) != 0:
        weights_t6 /= np.max(weights_t6)

    # Check the content of M_new[u] for debugging purposes
    print(f"M_new[u]: {M_new[u]}")
    r_node = 1
    ax.scatter(x_r4[:, 0], x_r4[:, 1], color="blue", marker=".", alpha=weights_t4, label=rf"$X_{{{u}}}^{{n,{4} \rightarrow {u}}}$")
    r_node = 3
    ax.scatter(x_r6[:, 0], x_r6[:, 1], color="purple", marker=".", alpha=weights_t6, label=rf"$X_{{{u}}}^{{n,{6} \rightarrow {u}}}$")
    # Loop over the neighbors
    # Loop over the neighbors
    for r in neighbors:
        if r != 4 and r != 6:
            continue
        
        r_kde = m_ru[(r, u)]
        Xr = r_kde.dataset.T[filtering:]
        Wr = r_kde.weights[filtering]
    
        
        # Plot particles with alpha values based on normalized weights


        pd_xru = pd.DataFrame(Xr, columns=["x", "y"])
        sns.kdeplot(data=pd_xru, x="x", y="y", common_norm=False,
                    fill=True,
                    weights=Wr,
                    bw_method='silverman',
                    bw_adjust=0.5,
                    color=colors[r % len(colors)],
                    alpha=0.2,
                    label=f"${{r}}$",
                    ax=ax)

        """ sns.kdeplot(data=pd_xru, x="x", y="y", common_norm=False,
                            fill=True,
                            weights=Wr,
                            bw_method='silverman',
                            bw_adjust=0.5,
                            color=colors[r % len(colors)],
                            alpha=0.2,
                            label=f"${{r}}$",
                            ax=axs[1]) """
        
    pd_xru = pd.DataFrame(M[u], columns=["x", "y"])
    sns.kdeplot(data=pd_xru, x="x", y="y", common_norm=False,
                fill=True,
                weights=weights[u],
                bw_method='silverman',
                bw_adjust=0.8,
                color='green',
                alpha=0.3,
                label=f"${{u}}$",
                ax=ax)
        
    ax.scatter(M[u, filtering:, 0], M[u, filtering:, 1], marker=".", c='green', alpha=0.4)
    #legend_handles.append(mpatches.Patch(color='green', label=rf"$M_{{{u}}}$"))
    
    #axs[0].set_title("Drawing particles from mixtures of messages")
    #ax.set_title("Drawing particles from mixtures of messages")
    #ax.set_title("2 Mixtures of messages")
    ax.set_title("Resampling with replacement")

    #axs[0].legend(handles=legend_handles)
    #axs[1].legend(handles=legend_handles)

    #ax.set_title("Product of 2 mixtures")
    ax.legend()
    plt.tight_layout()
    plt.show()

    """ kde_r1 = m_ru[(4, u)]
    kde_r2 = m_ru[(6, u)]

    xxr1 = kde_r1.dataset.T[filtering:]
    wwr1 = kde_r1.weights[filtering:]
    xxr2 = kde_r2.dataset.T[filtering:]
    wwr2 = kde_r2.weights[filtering:]
    
    new_samples = []
    for i in range(len(wwr1)):
        for j in range(len(wwr2)):
            new_sample = (xxr1[i] + xxr2[j]) / 2
            new_samples.append(new_sample)

    new_samples = np.array(new_samples).reshape(-1, 2)

    new_weights = np.ones(len(new_samples)) / len(new_samples)
    new_weights = new_weights[2*filtering**2:]
    new_samples = new_samples[2*filtering**2:]
    pd_xru_product = pandas.DataFrame(new_samples, columns=["x", "y"])
    sns.kdeplot(
            data=pd_xru_product,
            x="x",
            y="y",
            common_norm=False,
            fill=True,
            weights=new_weights,
            bw_method='silverman',
            bw_adjust=0.7,
            color='green',
            alpha=0.4,
            label="ru",
            ax=ax
        )
    ax.scatter(new_samples[:, 0], new_samples[:, 1], c="green", marker=".", label=r"$X_{"+ str(u) +"}$") """
    



def plot_representation(X: np.ndarray, anchors: np.ndarray, M: np.ndarray, weights: np.ndarray, node: int):
    n_anchors = anchors.shape[0]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(X[n_anchors:, 0], X[n_anchors:, 1], marker="o", c="gray", label="agents")
    ax.scatter(X[:n_anchors, 0], X[:n_anchors, 1], marker="*", c="red", label="references")
    ax.scatter(M[node, :, 0], M[node, :, 1], marker=".", c="green", label="Probable Locations for 5")

    pd_xru = pandas.DataFrame(M[node], columns=["x", "y"])
    sns.kdeplot(data=pd_xru, x="x", y="y", common_norm=False,
                fill=True,
                weights=weights[node],
                bw_method='silverman',
                color='green',
                alpha=0.2,
                label="PDF",
                ax=ax)
    
    for i, xr in enumerate(X):
        plt.annotate(f"{i}", xr)
    
    ax.set_title("A Probabilistic Representation")
    """ ax.set_ylim((30, 70))
    ax.set_xlim((5, 85)) """
    plt.tight_layout()
    plt.legend()
    plt.show()
    



def plot_compare_graphs(X, B, weighted_means, anchors, communication_radius, D, iteration, M=None, intersections=None, axs=None):
    """
    Compare true and estimated graphs.
    
    Parameters:
    - X: True positions of nodes.
    - weighted_means: Estimated positions of nodes.
    - anchors: Positions of anchor nodes.
    - communication_radius: Communication radius for the network.
    - D: Adjacency matrix of the network.
    - iteration: Current iteration of the algorithm.
    """
    n_anchors = anchors.shape[0]
    estimates = np.concatenate([anchors, weighted_means])

    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot true graph
    #axs[0].scatter(X[:n_anchors, 0], X[:n_anchors, 1], marker="*", c="red")
    #nx.draw_networkx_edges(nx.from_numpy_array(D), pos=X, ax=axs[0], width=0.5)
    #axs[0].set_title("True Graph")
    axs[0].set_title("Particles from Previous iteration")
    """ if M is not None:
        plot_initial_particles(X, D, M, anchors, intersections, ax=axs[0])
    else: """
    axs[0].scatter(X[:n_anchors, 0], X[:n_anchors, 1], marker="*", c="red")
    nx.draw_networkx_edges(nx.from_numpy_array(D * B), pos=X, ax=axs[0], width=0.5)
    axs[0].set_title("True Graph")

    # Plot estimated graph
    axs[1].scatter(estimates[:n_anchors, 0], estimates[:n_anchors, 1], marker="*", c="red")
    nx.draw_networkx_edges(nx.from_numpy_array(D * B), pos=estimates, ax=axs[1], width=0.5)
    axs[1].set_title("Estimated Graph")
    plt.tight_layout()
    plt.show()
    return axs


def plot_results(M, weights, X, predicts, anchors, intersections, iteration, show_bbox=False, uncertainties=None, ax=None):
    """
    Plot the results of the localization algorithm using a single DataFrame.
    
    Parameters:
    - M: Particles for each node.
    - weights: Weights for each particle.
    - X: True positions of nodes.
    - predicts: Predicted positions of nodes.
    - anchors: Positions of anchor nodes.
    - intersections: Bounding boxes for non-anchor nodes.
    - iteration: Current iteration of the algorithm.
    - show_bbox: Flag to show bounding boxes.
    """
    n_anchors = anchors.shape[0]
    n_samples = X.shape[0]

    sns.set(style="whitegrid")
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Create a single DataFrame for all particles
    all_particles = []
    all_weights = []
    nodes = []
    for i, (Xr, Wr) in enumerate(zip(M, weights)):
        if i < n_anchors:  # Skip anchors
            continue
        all_particles.extend(Xr)
        all_weights.extend(Wr)
        nodes.extend([f"Node {i}"] * len(Wr))

    """ df_particles = pandas.DataFrame({
        'x': [p[0] for p in all_particles],
        'y': [p[1] for p in all_particles],
        'weights': all_weights,
        'node': nodes
    }) """

    # Plot bounding boxes if required
    if show_bbox:
        for counter in range(n_anchors, n_samples):
            bbox = intersections[counter]
            xmin, xmax, ymin, ymax = bbox
            ax.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], '-')

    # Plot anchors, true positions, and predictions
    ax.scatter(anchors[:, 0], anchors[:, 1], label=rf"$N_{{a}}$", c="red", marker='*')
    ax.scatter(X[n_anchors:, 0], X[n_anchors:, 1], label=rf"$N_{{t}}$", c="green", marker='P')
    ax.scatter(predicts[:, 0], predicts[:, 1], label="Estimates", c="orange", marker="X")
    ax.plot([X[n_anchors:, 0], predicts[:, 0]], [X[n_anchors:, 1], predicts[:, 1]], "k--")

    """ colors = ['cyan', 'green', 'pink', 'yellow']
    # KDE plot for all particles using the single DataFrame
    sns.kdeplot(data=df_particles, x="x", y="y", weights="weights", hue="node",
                fill=True, common_norm=False, alpha=0.3, ax=ax) """

    # Annotate node numbers
    for j, p in enumerate(X):
        if j < n_anchors:
            ax.annotate(rf"$A_{{{j}}}$", p)
        else:
            ax.annotate(rf"$T_{{{j}}}$", p)

    # Draw uncertainty circles if uncertainties are provided
    if uncertainties is not None:
        for i, (pred, uncertainty) in enumerate(zip(predicts, uncertainties)):
            circle = plt.Circle((pred[0], pred[1]), uncertainty, color='blue', fill=False, linestyle='--')
            ax.add_artist(circle)
            ax.annotate(f"U_{i}", (pred[0], pred[1]), color='blue')

    # Set plot properties
    ax.set_title("Localization Results")
    #plt.ylim((30, 70))
    #plt.xlim((3, 88))
    ax.legend()
    #plt.tight_layout()
    #if iteration is not None:
    #    plt.savefig(f"{folder}/figure_iter_{iteration}_{hop}.png")
    #plt.show()
    return ax

def plot_procrustes(similarities, iter, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    top = 1  # Get the maximum RMSE value
    bottom = 0
    ax.set_ylim((bottom, top))  # Set the y-axis limits
    ax.plot(np.arange(iter), similarities, label="Similarity")  # Plot only the RMSE values
    ax.set_ylabel("Similarity")
    ax.set_xlabel("iteration")
    #plt.tight_layout()
    ax.legend()
    #plt.savefig(f"{folder}/rmseplot.png")
    #plt.show()
    return ax

def plot_simple_once(X, anchors, D, weighted_means):
    fig, axs = plt.subplots(1, 2, figsize=(12, 8) ,sharex=True, sharey=True)
    estimates = np.concatenate([anchors, weighted_means])
    n_anchors = anchors.shape[0]
    axs[0].scatter(anchors[:, 0], anchors[:, 1], label="Anchors", c="red", marker='*')
    axs[0].scatter(X[n_anchors:, 0], X[n_anchors:, 1], label="True Positions", c="green", marker='P')
    nx.draw_networkx_edges(nx.from_numpy_array(D), pos=X, ax=axs[0], width=0.5)

    axs[1].scatter(anchors[:, 0], anchors[:, 1], label="Anchors", c="red", marker='*')
    axs[1].scatter(estimates[n_anchors:, 0], estimates[n_anchors:, 1], label="Predictions", c="orange", marker='P')
    nx.draw_networkx_edges(nx.from_numpy_array(D), pos=estimates, ax=axs[1], width=0.5)
    
    plt.tight_layout()
    axs[1].set_title("Estimates")
    axs[0].set_title("True")
    plt.show()

def error_vs_neighborhood(n_anchors: int, D: np.ndarray, estimates: np.ndarray, trues: np.ndarray):
    one_neighbour_count = [np.count_nonzero(u > 0) for u in D]
    errors = ERR(estimates, trues)
    z = np.polyfit(one_neighbour_count[n_anchors:], errors, 2)
    p = np.poly1d(z)
    return p


def plot_network_error(weighted_means, X, anchors, D, B, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    n_anchors = anchors.shape[0]
    one_neighbour_count = [np.count_nonzero(u > 0) for u in D*B]
    #anchor_neighbours = [np.count_nonzero(u[anchors.shape[0]] > 0) for u in D*B]
    errors = ERR(weighted_means, X[n_anchors:])
    # Create a scatter plot of neighbour count vs error
    ax.scatter(one_neighbour_count[n_anchors:], errors,  alpha=0.6, color="blue")

    correlation = np.corrcoef(one_neighbour_count[n_anchors:], errors)[0, 1]
    p_val = correlation_p_value(correlation, weighted_means.shape[0])
    print(f"Cor: {correlation}, p-val: {p_val}")
    # Fit a polynomial curve of degree 2 to the data
    z = np.polyfit(one_neighbour_count[n_anchors:], errors, 2)
    p = np.poly1d(z)

    # Plot the curve on the same figure
    x = np.linspace(min(one_neighbour_count[n_anchors:]), max(one_neighbour_count[n_anchors:]), 100)
    ax.plot(x, p(x), color='red', label="one hop")
    # Add labels and title
    ax.set_xlabel('Neighbour count')
    ax.set_ylabel('Error')
    ax.set_title('Neighbour count vs error with polynomial fit')

    #plt.tight_layout()
    #plt.show()
    return ax

def plot_MRF(X: np.ndarray, B: np.ndarray, n_anchors: int, network: np.ndarray, radius: int, intersections: np.ndarray):
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    G = nx.from_numpy_array(network*B)

    # Plot edges between node of interest and its immediate neighbors
    pos = {i: (X[i][0], X[i][1]) for i in range(len(X))}
    nx.draw_networkx_edges(G, pos=pos, ax=ax, edge_color='gray')
    

    th = 0
    # Highlight intersection areas
    """ for i in range(n_anchors, len(X)):
        if intersections is not None and i < len(intersections):
            if B[i, :n_anchors].sum() > th:
                #if network[i, :n_anchors].sum() > 0:
                int_bbox = intersections[i]
                xmin, xmax, ymin, ymax = int_bbox
                ax.fill_between([xmin, xmax], ymin, ymax, alpha=0.3) """
        

    # Plot targets
    for i in range(0, n_anchors):
        if intersections is not None:
            int_bbox = intersections[i]
            xmin, xmax, ymin, ymax = int_bbox
            ax.fill_between([xmin, xmax], ymin, ymax, alpha=0.3)

    # Plot anchors
    ax.scatter(X[:n_anchors, 0], X[:n_anchors, 1], marker="*", c="r", label=r"$N_{a}$", s=300)
    for i in range(n_anchors):
        ax.annotate(rf"$A_{{{i}}}$", (X[i, 0], X[i, 1]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=12, color='r')


    plotted = False
    for i in range(n_anchors, len(X)):
        if B[i, :n_anchors].sum() > th:
            if not plotted:
                ax.scatter(X[i, 0], X[i, 1], marker="+", c="g", label=r"$N_{t}$", s=75)
                plotted = True
            else:
                 ax.scatter(X[i, 0], X[i, 1], marker="+", c="g", s=75)
            ax.annotate(rf"$T_{{{i - n_anchors}}}$", (X[i, 0], X[i, 1]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=12, color='g')
    
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.title("Network Coverage")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_initial_particles_MRF(X: np.ndarray, all_particles: np.ndarray, weights: np.ndarray,  n_anchors: int, network: np.ndarray, radius: int, intersections: np.ndarray):
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    x_lim = (5, 82)
    y_lim = (34, 65)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    G = nx.from_numpy_array(network)

    one_hop = network.copy()
    one_hop[one_hop > radius] = 0
    G1 = nx.from_numpy_array(one_hop)

    two_hop = network.copy()
    two_hop[two_hop < radius] = 0
    G2 = nx.from_numpy_array(two_hop)

    handles = []
    pos = {i: (X[i][0], X[i][1]) for i in range(len(X))}
    filter_particles = 75
    colors = ["red", "blue", "green", "yellow"]
    for i in range(n_anchors, len(X)):
        # Create a DataFrame with the coordinates

        x_p = all_particles[i][filter_particles:]
        w_p = weights[i][filter_particles:]
        pd_xru = pandas.DataFrame(x_p, columns=["x", "y"])
        ax.scatter(x_p[:, 0], x_p[:, 1], c=colors[i - n_anchors], marker=".")

        # Plot the KDE
        sns.kdeplot(data=pd_xru, x="x", y="y", common_norm=False,
                    fill=True,
                    weights=w_p,
                    bw_method='silverman',
                    bw_adjust=0.8,
                    color=colors[i - n_anchors],
                    alpha=0.3,
                    label=f"u",
                    common_grid=True
                    )
        
        kde_handle = mpatches.Patch(color=colors[i - n_anchors], label=f'$b_{{{i}}}^{{{1}}}$')
        handles.append(kde_handle)
    
        """ if intersections is not None and i < len(intersections):
            int_bbox = intersections[i]
            xmin, xmax, ymin, ymax = int_bbox
            plt.fill_between([xmin, xmax], ymin, ymax, color=colors[i - n_anchors], alpha=0.3, label=f"Prior of $\psi_{{{i}}}(x_{{{i}}})$") """
            

    # Plot anchor nodes in red
    anchor_nodes = range(n_anchors)
    nx.draw_networkx_nodes(G, pos=pos, nodelist=anchor_nodes, node_color='red', node_shape='o', node_size=400)

    # Plot non-anchor nodes in blue
    non_anchor_nodes = range(n_anchors, len(X))
    #nx.draw_networkx_nodes(G, pos=pos, nodelist=non_anchor_nodes, node_color='blue', node_shape='o', node_size=600, label=r"$N_{t}$")

    # Add labels to the nodes
    node_labels = {i: f'$(x_{{{i}}})$' for i in range(n_anchors)}
    nx.draw_networkx_labels(G, pos=pos, labels=node_labels, font_size=12, font_color='white')

    # Add specific labels for non-anchor nodes
    offset = -5
    for i in non_anchor_nodes:
        if X[i][1] <= 50:
            ax.text(X[i][0], X[i][1] - offset, f'$b_{{{i}}}^{{{1}}}$', horizontalalignment='center', verticalalignment='top', fontsize=14, color='black')
        else:
            ax.text(X[i][0], X[i][1] + offset, f'$b_{{{i}}}^{{{1}}}$', horizontalalignment='center', verticalalignment='bottom', fontsize=14, color='black')

    ax.set_title("Two-hop message approximations", fontsize=12)
    #ax.axis('off')  # Remove the axis

    plt.legend(handles=handles, fontsize=12, loc='lower right')
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

def plot_belief_update(all_particles: np.ndarray, messages: dict, node_of_interest: int):
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))

    colors = ['blue', 'red', 'green', 'yellow', 'purple', 'cyan', 'orange', 'pink']
    neighbor_labels = {0: "m_{0 \rightarrow 4}", 1: "m_{1 \rightarrow 4}", 5: "m_{5 \rightarrow 4}", 7: "m_{7 \rightarrow 4}", 4: "b_4"}

    # Plot the KDE for the node of interest
    x_p = all_particles[node_of_interest]
    pd_node = pd.DataFrame(x_p, columns=["x", "y"])
    
    sns.kdeplot(data=pd_node, x="x", y="y", common_norm=False,
                fill=True,
                color="black",
                bw_method='silverman',
                bw_adjust=0.5,
                alpha=0.5,
                label=r"$X_{4}^{n-1}$")
    
    kde_handles = []
    for idx, (neighbor, label) in enumerate(neighbor_labels.items()):
        if neighbor == node_of_interest:
            continue
        
        kde_ru = messages[neighbor, node_of_interest]
        x_ru = kde_ru.dataset.T
        pd_xru = pd.DataFrame(x_ru, columns=["x", "y"])

        color = colors[idx % len(colors)]
        sns.kdeplot(data=pd_xru, x="x", y="y", common_norm=False,
                    fill=True,
                    color=color,
                    bw_method='silverman',
                    bw_adjust=0.5,
                    alpha=0.5,
                    label=rf"${label}$")
        
        handle = mpatches.Patch(color=color, label=rf"${label}$")
        kde_handles.append(handle)

    plt.title("Mixtures of Gaussians for Messages to Node 4")
    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    plt.show()


def plot_all_messages(D: np.ndarray,
                      B: np.ndarray,
                      radius: int,
                      m_ru: dict,
                      node_of_interest: int,
                      all_particles: np.ndarray,
                      weights: np.ndarray,
                      X: np.ndarray,
                      n_anchors: int):
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    # Create the graph from the distance matrix
    G = nx.from_numpy_array(D)

    # Create a graph with only one-hop neighbors
    one_hop = D.copy() * B
    one_hop[one_hop > radius] = 0
    G1 = nx.from_numpy_array(one_hop)

    # Get neighbors of the node of interest
    neighbors = list(G1.neighbors(node_of_interest))
    print("One-hop neighbors:", neighbors)

    # Find two-hop neighbors
    two_hop = D.copy() * ((B - 1) * (B - 1))
    #two_hop[two_hop < radius] = 0
    G2 = nx.from_numpy_array(two_hop)
    two_hop_neighbors = list(G2.neighbors(node_of_interest))
    two_hop_neighbors = [n for n in two_hop_neighbors if n not in neighbors and n != node_of_interest]
    print("Two-hop neighbors:", two_hop_neighbors)

    filter_particles = 50
    kde_handles = []
    
    colors = ['blue', 'red', 'orange', 'purple', 'cyan', 'green', 'pink', 'yellow']
    node_colors = {}
    
    for neighbor in neighbors:
        # Retrieve KDE and particle data for the neighbor
        kde_ru = m_ru[neighbor, node_of_interest]
        x_ru = kde_ru.dataset.T
        w_ru = kde_ru.weights

        pd_xru = pd.DataFrame(x_ru, columns=["x", "y"])
        color = colors[neighbor % len(colors)]
        node_colors[neighbor] = color
        
        # Scatter plot of particles
        sns.kdeplot(data=pd_xru, x="x", y="y", common_norm=False,
                    fill=True,
                    weights=w_ru,
                    color=color,
                    bw_method='silverman',
                    bw_adjust=0.4,
                    alpha=0.4)
        
        handle = mpatches.Patch(color=color, label=rf"$m^{{n}}_{{{neighbor} \rightarrow {node_of_interest}}}$")
        kde_handles.append(handle)
    
    # Plot the KDE for the node of interest
    x_p = all_particles[node_of_interest]
    w_p = weights[node_of_interest]
    pd_node = pd.DataFrame(x_p, columns=["x", "y"])
    
    # Scatter plot of particles
    ax.scatter(x_p[filter_particles:, 0], x_p[filter_particles:, 1], c="black", marker=".", label=rf"$X^{{n-1}}_{{{node_of_interest}}}$")
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend(kde_handles)
    sns.kdeplot(data=pd_node, x="x", y="y", common_norm=False,
                fill=True,
                weights=w_p,
                bw_method='silverman',
                bw_adjust=0.5,
                color="black",
                alpha=0.8,
                label=rf"$b^n_{{{node_of_interest}}}$",
                common_grid=True)
    
    target_handle = mpatches.Patch(color="black", label=rf"$b^{{n-1}}_{{{node_of_interest}}}$")
    handles.append(target_handle)
    
    # Plot edges between node of interest and its immediate neighbors
    pos = {i: (X[i][0], X[i][1]) for i in range(len(X))}
    nx.draw_networkx_edges(G1, pos=pos, edgelist=[(node_of_interest, neighbor) for neighbor in neighbors], ax=ax, edge_color='gray', label=rf"$c_{{ij}} = 1$")
    
    # Plot two-hop edges
    nx.draw_networkx_edges(G2, pos=pos, edgelist=[(node_of_interest, neighbor) for neighbor in two_hop_neighbors], ax=ax, edge_color='blue', style='dashed', label=rf"$c_{{ij}} = 1$")

    # Define anchor nodes (first 4 elements) and non-anchor nodes
    anchor_nodes = list(range(4))
    non_anchor_nodes = list(range(4, len(X)))
    
    # Plot anchor nodes in red
    nx.draw_networkx_nodes(G1, pos=pos, nodelist=anchor_nodes, node_color='red', node_size=400, ax=ax)
    # Plot non-anchor nodes in blue, except those with message distributions
    non_anchor_nodes = [node for node in non_anchor_nodes if node not in node_colors]
    nx.draw_networkx_nodes(G1, pos=pos, nodelist=non_anchor_nodes, node_size=400, ax=ax)
    
    # Plot nodes with message distribution colors
    for node, color in node_colors.items():
        nx.draw_networkx_nodes(G1, pos=pos, nodelist=[node], node_color=color, node_size=400, ax=ax)
    
    # Add labels to nodes
    labels = {i: rf'$A_{{{i}}}$' if i < 4 else f'$T_{{{i - n_anchors}}}$' for i in range(len(X))}
    nx.draw_networkx_labels(G1, pos=pos, labels=labels, font_size=12, font_color='white', ax=ax)
    
    plt.legend(handles=handles, fontsize=12, loc='upper right')
    plt.title(f"Mixtures of Gaussians for messages to {node_of_interest}")
    plt.tight_layout()
    plt.show()


def plot_sample_particles(X: np.ndarray, all_particles: np.ndarray,
                           weights: np.ndarray, n_anchors: int,
                           network: np.ndarray, radius: int,
                           intersections: np.ndarray, x_ru: np.ndarray,
                           u: int, r: int):
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    x_lim = (5, 82)
    y_lim = (34, 65)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    G = nx.from_numpy_array(network)

    one_hop = network.copy()
    one_hop[one_hop > radius] = 0
    G1 = nx.from_numpy_array(one_hop)

    two_hop = network.copy()
    two_hop[two_hop < radius] = 0
    G2 = nx.from_numpy_array(two_hop)

    # Define the bounds of the box for the uniform distribution
    x_min, x_max, y_min, y_max = 0, 85, 5, 80
    
    # Create a uniform grid of points within the bounded box
    x = np.linspace(x_min, x_max, 400)
    y = np.linspace(y_min, y_max, 400)
    XX, YY = np.meshgrid(x, y)
    grid_points = np.vstack([XX.ravel(), YY.ravel()]).T

    """ probabilities = detection_probability(grid_points, X[u], radius, network[r, u])
    Z = probabilities.reshape(XX.shape)
    cp = ax.contourf(XX, YY, Z, alpha=0.3, cmap='viridis', levels=25) """

    """ pd_xru = pd.DataFrame(x_ru, columns=["x", "y"])
    # Plot the KDE
    sns.kdeplot(data=pd_xru, x="x", y="y", common_norm=False,
                fill=True,
                bw_method='silverman',
                bw_adjust=0.42,
                color="pink",
                alpha=0.4,
                label=f"u",
                common_grid=True
                ) """

    handles = []
    pos = {i: (X[i][0], X[i][1]) for i in range(len(X))}
    filter_particles = 75
    

    colors = ["red", "blue", "green", "yellow"]
    for i in range(n_anchors, len(X)):
        # Create a DataFrame with the coordinates
        x_p = all_particles[i]
        w_p = weights[i]
        pd_xru = pd.DataFrame(x_p, columns=["x", "y"])

        """ if i == u or i == r:
            ax.scatter(x_p[filter_particles:, 0], x_p[filter_particles:, 1], c=colors[i - n_anchors], marker=".", label=rf"$X^{{n-1}}_{{{i}}}$")
        else:
            ax.scatter(x_p[filter_particles:, 0], x_p[filter_particles:, 1], c=colors[i - n_anchors], marker=".") """

        # Plot the KDE
        sns.kdeplot(data=pd_xru, x="x", y="y", common_norm=False,
                    fill=True,
                    weights=w_p,
                    bw_method='silverman',
                    bw_adjust=0.8,
                    color=colors[i - n_anchors],
                    alpha=0.3,
                    label=f"u",
                    common_grid=True
                    )

    # Plot anchor nodes in red
    anchor_nodes = range(n_anchors)
    nx.draw_networkx_nodes(G, pos=pos, nodelist=anchor_nodes, node_color='red', node_shape='o', node_size=400)

    # Plot non-anchor nodes in blue
    non_anchor_nodes = range(n_anchors, len(X))
    # nx.draw_networkx_nodes(G, pos=pos, nodelist=non_anchor_nodes, node_color='blue', node_shape='o', node_size=600, label=r"$N_{t}$")

    # Add labels to the nodes
    node_labels = {i: f'$(x_{{{i}}})$' for i in range(n_anchors)}
    nx.draw_networkx_labels(G, pos=pos, labels=node_labels, font_size=12, font_color='white')

    # Add specific labels for non-anchor nodes
    offset = -5
    for i in non_anchor_nodes:
        if X[i][1] <= 50:
            ax.text(X[i][0], X[i][1] - offset, f'$b_{{{i}}}^{{n}}$', horizontalalignment='center', verticalalignment='top', fontsize=14, color='black')
        else:
            ax.text(X[i][0], X[i][1] + offset, f'$b_{{{i}}}^{{n}}$', horizontalalignment='center', verticalalignment='bottom', fontsize=14, color='black')

    # Draw an empty circle at the location X[u] with radius as network[u, r]
    circle = Circle((X[r][0], X[r][1]), network[r, u], edgecolor='black', facecolor='none', linestyle='--')
    ax.add_patch(circle)

    # Draw a line between X[u] and X[r]
    line = Line2D([X[r][0], X[u][0]], [X[r][1], X[u][1]], color='black', linestyle='--')
    ax.add_line(line)

    # Display a label on the middle of the line depicting "d_ij"
    mid_x = (X[u][0] + X[r][0]) / 2
    mid_y = (X[u][1] + X[r][1]) / 2
    ax.text(mid_x, mid_y, r'$d_{ij}$', fontsize=12, color='black', horizontalalignment='center')

    #ax.set_title(r"Message approximation for $m^{n}_{4 \rightarrow 5}$", fontsize=12)
    ax.set_title(r"Obtaining angular particles $X^{n}_{4 \rightarrow 5}$ of message $m^{n}_{4 \rightarrow 5}$", fontsize=12)
    #patch_4 = mpatches.Patch(color="pink", label=r'message $m^{n}_{4 \rightarrow 5}$', alpha=0.75)
    ax.scatter(x_ru[:, 0], x_ru[:, 1], c="pink", marker="*", label=r"$X^{n}_{4 \rightarrow 5}$")
    handles, labels = ax.get_legend_handles_labels()
    #handles.append(patch_4)
    plt.legend(fontsize=12, loc='lower right', handles=handles)
    plt.tight_layout()
    plt.show()

def plot_message_particles(X: np.ndarray, all_particles: np.ndarray,
                           weights: np.ndarray, n_anchors: int,
                           network: np.ndarray, radius: int,
                           intersections: np.ndarray, x_ru: np.ndarray,
                           u: int, r: int):
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(9,5))
    """ x_lim = (5, 82)
    y_lim = (34, 65)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim) """
    G = nx.from_numpy_array(network)

    one_hop = network.copy()
    one_hop[one_hop > radius] = 0
    G1 = nx.from_numpy_array(one_hop)

    two_hop = network.copy()
    two_hop[two_hop < radius] = 0
    G2 = nx.from_numpy_array(two_hop)

    # Define the bounds of the box for the uniform distribution
    x_min, x_max, y_min, y_max = 0, 85, 5, 80
    
    # Create a uniform grid of points within the bounded box
    x = np.linspace(x_min, x_max, 400)
    y = np.linspace(y_min, y_max, 400)
    XX, YY = np.meshgrid(x, y)
    grid_points = np.vstack([XX.ravel(), YY.ravel()]).T

    """ probabilities = detection_probability(grid_points, X[u], radius, network[r, u])
    Z = probabilities.reshape(XX.shape)
    cp = ax.contourf(XX, YY, Z, alpha=0.3, cmap='viridis', levels=25) """

    pd_xru = pd.DataFrame(x_ru, columns=["x", "y"])
    # Plot the KDE
    sns.kdeplot(data=pd_xru, x="x", y="y", common_norm=False,
                fill=True,
                bw_method='silverman',
                bw_adjust=0.6,
                color="red",
                alpha=0.5,
                label=f"u"
                )

    handles = []
    pos = {i: (X[i][0], X[i][1]) for i in range(len(X))}
    filter_particles = 75
    ax.scatter(x_ru[filter_particles:, 0], x_ru[filter_particles:, 1], c="red", marker=".", label=r"$X^{n}_{4 \rightarrow 5}$")

    colors = ["red", "blue", "green", "yellow"]
    for i in range(n_anchors, len(X)):
        if i != u and i != r:
            continue
        # Create a DataFrame with the coordinates
        x_p = all_particles[i]
        w_p = weights[i]
        pd_xru = pd.DataFrame(x_p, columns=["x", "y"])

        
        ax.scatter(x_p[filter_particles:, 0], x_p[filter_particles:, 1], c=colors[i - n_anchors], marker=".", label=rf"$X^{{n-1}}_{{{i}}}$")
    

        # Plot the KDE
        sns.kdeplot(data=pd_xru, x="x", y="y", common_norm=False,
                    fill=True,
                    weights=w_p,
                    bw_method='silverman',
                    bw_adjust=0.8,
                    color=colors[i - n_anchors],
                    alpha=0.3,
                    label=f"u",
                    common_grid=True
                    )

    # Plot anchor nodes in red
    anchor_nodes = range(n_anchors)
    nx.draw_networkx_nodes(G, pos=pos, nodelist=anchor_nodes, node_color='red', node_shape='o', node_size=400)

    # Plot non-anchor nodes in blue
    non_anchor_nodes = range(n_anchors, len(X))
    # nx.draw_networkx_nodes(G, pos=pos, nodelist=non_anchor_nodes, node_color='blue', node_shape='o', node_size=600, label=r"$N_{t}$")

    # Add labels to the nodes
    node_labels = {i: f'$(x_{{{i}}})$' for i in range(n_anchors)}
    nx.draw_networkx_labels(G, pos=pos, labels=node_labels, font_size=12, font_color='white')

    # Add specific labels for non-anchor nodes
    offset = -5
    for i in non_anchor_nodes:
        if X[i][1] <= 50:
            ax.text(X[i][0], X[i][1] - offset, f'$b_{{{i}}}^{{n}}$', horizontalalignment='center', verticalalignment='top', fontsize=14, color='black')
        else:
            ax.text(X[i][0], X[i][1] + offset, f'$b_{{{i}}}^{{n}}$', horizontalalignment='center', verticalalignment='bottom', fontsize=14, color='black')

    """ # Draw an empty circle at the location X[u] with radius as network[u, r]
    circle = Circle((X[r][0], X[r][1]), network[r, u], edgecolor='black', facecolor='none', linestyle='--')
    ax.add_patch(circle)

    # Draw a line between X[u] and X[r]
    line = Line2D([X[r][0], X[u][0]], [X[r][1], X[u][1]], color='black', linestyle='--')
    ax.add_line(line)

    # Display a label on the middle of the line depicting "d_ij"
    mid_x = (X[u][0] + X[r][0]) / 2
    mid_y = (X[u][1] + X[r][1]) / 2
    ax.text(mid_x, mid_y, r'$d_{ij}$', fontsize=12, color='black', horizontalalignment='center') """

    ax.set_title(r"Message approximation for $m^{n}_{4 \rightarrow 5}$", fontsize=12)
    patch_4 = mpatches.Patch(color="pink", label=r'message $m^{n}_{4 \rightarrow 5}$', alpha=0.75)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(patch_4)
    plt.legend(fontsize=12, loc='lower right', handles=handles)
    plt.tight_layout()
    plt.show()



def plot_MRF_with_detailed_messages(X, network, node_of_interest):
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    G = nx.from_numpy_array(network)
    pos = {i: (X[i][0], X[i][1]) for i in range(len(X))}
    
    # Plot the edges
    nx.draw_networkx_edges(G, pos=pos, ax=ax, width=2)

    # Plot the nodes with Seaborn style
    nx.draw_networkx_nodes(G, pos=pos, node_shape='o', node_size=1400)

    # Add labels to the nodes
    labels = {i: f'x_{i}' for i in range(len(X))}
    nx.draw_networkx_labels(G, pos=pos, labels=labels, font_size=12, font_color='white')

    # Highlight the node of interest
    nx.draw_networkx_nodes(G, pos=pos, nodelist=[node_of_interest], node_color='red', node_size=1400)

    # Add arrows to represent messages
    neighbors = list(G.neighbors(node_of_interest))
    label_offset = 0.2  # Offset for label placement to avoid overlap

    for neighbor in neighbors:
        other_neighbors = list(G.neighbors(neighbor))
        other_neighbors.remove(node_of_interest)  # Exclude the node of interest

        for i, other in enumerate(other_neighbors):
            ax.annotate("",
                        xy=pos[neighbor], xycoords='data',
                        xytext=pos[other], textcoords='data',
                        arrowprops=dict(arrowstyle="->", color='orange', lw=3))
            # Add message label with LaTeX formatting for subscripts
            mid_point = ((pos[neighbor][0] + pos[other][0]) / 2 + label_offset * (i + 1) + np.random.normal(0, 0.8),
                         (pos[neighbor][1] + pos[other][1]) / 2 + label_offset * (i + 1) + np.random.normal(0, 0.8))
            ax.text(mid_point[0], mid_point[1], f'$m_{{{other} \\to {neighbor}}}$', fontsize=12, color='black', ma='center')

    # Add \phi(x_{i}, x_{7}) labels on edges connected to node_of_interest
    for neighbor in neighbors:
        edge_mid_point = ((pos[node_of_interest][0] + pos[neighbor][0]) / 2 ,
                          (pos[node_of_interest][1] + pos[neighbor][1]) / 2 )
        ax.text(edge_mid_point[0], edge_mid_point[1], f'$\\psi(x_{{{neighbor}}}, x_{{{node_of_interest}}})$', fontsize=12, color='black', ha='center', ma='right')

    # Add arrows from neighbors to node_of_interest in a different color
    for neighbor in neighbors:
        ax.annotate("",
                    xy=pos[node_of_interest], xycoords='data',
                    xytext=pos[neighbor], textcoords='data',
                    arrowprops=dict(arrowstyle="->", color='green', lw=2))

    # Add \phi(x_{i}) labels next to neighbors of node_of_interest
    phi_offset = 0.2  # Offset for phi label placement
    for neighbor in neighbors:
        phi_label_position = (pos[neighbor][0], pos[neighbor][1] + phi_offset)
        ax.text(phi_label_position[0], phi_label_position[1]+2, f'$\\psi(x_{{{neighbor}}})$', fontsize=12, color='black', ha='center')


    plt.axis('off')  # Remove the axis
    plt.tight_layout()
    plt.show()

def plot_proposal_distributions(X: np.ndarray, node: int, all_particles: np.ndarray, network: np.ndarray, m_ru, M_new):
    # Set a professional style
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=100)
    size = 500
    
    # Create graph from network adjacency matrix
    G = nx.from_numpy_array(network)
    pos = {i: (X[i][0], X[i][1]) for i in range(len(X))}
    colors = ['blue', 'red', 'orange', 'purple', 'cyan', 'green', 'pink', 'yellow']
    neighbors = list(G.neighbors(node))
    cc = 0
    ppp = 222
    for neighbor in neighbors:
        if cc == 2:
            break
        cc += 1
        r_kde = m_ru[(neighbor, node)].dataset.T
        pd_xru = pandas.DataFrame(r_kde, columns=["x", "y"])
        sns.kdeplot(data=pd_xru, x="x", y="y", common_norm=False,
                    fill=True,
                    bw_method='silverman',
                    gridsize=100,
                    cbar=False,
                    color=colors[neighbor],
                    alpha=0.18,
                    ax=ax,
                    bw_adjust=0.5)
        
        ax.scatter(M_new[node][neighbor-3][ppp:, 0], M_new[node][neighbor-3][ppp:, 1], marker=".", c=colors[neighbor], label = f"particles sampled from $q_{{{neighbor,node}}}$", alpha=0.3)
    
    # Plot the edges of the graph
    nx.draw_networkx_edges(G, pos=pos, ax=ax, width=1, alpha=0.5, edge_color='grey')

    # Plot the nodes with Seaborn style
    nx.draw_networkx_nodes(G, pos=pos, node_shape='o', node_size=size, alpha=0.5, node_color='skyblue', ax=ax)

    # Add labels to the nodes with LaTeX formatting for subscripts
    labels = {i: f'$X_{{{i}}}$' for i in range(len(X))}
    nx.draw_networkx_labels(G, pos=pos, labels=labels, font_size=12, font_color='black', ax=ax)

    # Highlight the node of interest
    nx.draw_networkx_nodes(G, pos=pos, nodelist=[node], node_color='red', node_size=size, alpha=0.6, ax=ax)

    # Plot particles
    aa = 0.4
    ax.scatter(all_particles[node, ppp-50:, 0], all_particles[node, ppp-50:, 1], label=f"particles from posterior of $X_{{{node}}}$", c="red", alpha=aa)

    """ # Create custom legend
    patch_3 = mpatches.Patch(color=colors[3], label=f'proposal $q_{{{3,node}}}$', alpha=0.18)
    patch_4 = mpatches.Patch(color=colors[4], label=f'proposal $q_{{{4,node}}}$', alpha=0.18)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(patch_3)
    handles.append(patch_4)
    ax.legend(handles=handles, fontsize=10) """
    ax.legend()

    # Remove the axis for cleaner look
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def plot_particle_filter(X: np.ndarray, node: int, all_particles: np.ndarray, network: np.ndarray):
    # Set a professional style
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=100)
    size = 800
    
    # Convert particles to DataFrame for KDE plotting
    pd_xru = pandas.DataFrame(all_particles[node], columns=["x", "y"])
    sns.kdeplot(data=pd_xru, x="x", y="y", common_norm=False,
                fill=True,
                bw_method='silverman',
                gridsize=100,
                cbar=False,
                color="blue",
                alpha=0.5,
                ax=ax)
    
    # Create graph from network adjacency matrix
    G = nx.from_numpy_array(network)
    pos = {i: (X[i][0], X[i][1]) for i in range(len(X))}
    
    # Plot the edges of the graph
    nx.draw_networkx_edges(G, pos=pos, ax=ax, width=1, alpha=0.5, edge_color='grey')

    # Plot the nodes with Seaborn style
    nx.draw_networkx_nodes(G, pos=pos, node_shape='o', node_size=size, alpha=0.5, node_color='skyblue')

    # Add labels to the nodes with LaTeX formatting for subscripts
    labels = {i: f'$X_{{{i}}}$' for i in range(len(X))}
    nx.draw_networkx_labels(G, pos=pos, labels=labels, font_size=12, font_color='black')

    # Highlight the node of interest
    nx.draw_networkx_nodes(G, pos=pos, nodelist=[node], node_color='red', node_size=size, alpha=0.6)

    # Plot particles
    ppp = 135
    aa = 0.4
    ax.scatter(all_particles[node, ppp:, 0], all_particles[node, ppp:, 1], label=f"particles from posterior of $X_{{{node}}}$", c="red", alpha=aa)

    # Create custom legend
    blue_patch = mpatches.Patch(color='blue', label=f'posterior PDF of RV $X_{{{node}}}$', alpha=0.4)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(blue_patch)
    ax.legend(handles=handles, loc=(0.7, 0.25), fontsize=10)
    
    # Add title and annotations
    ax.set_title('Particle Filtering Visualization', fontsize=16)

    # Remove the axis for cleaner look
    plt.axis('off')
    plt.tight_layout()
    plt.show()



def plot_Bayesian_Network(X: np.ndarray, a: int, network):
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    # Create a directed graph
    G = nx.DiGraph()
    edges = np.argwhere(network > 0)
    
    # Add edges ensuring the graph remains acyclic
    for edge in edges:
        G.add_edge(edge[0], edge[1])
        if not nx.is_directed_acyclic_graph(G):
            G.remove_edge(edge[0], edge[1])
    
    # Plot the directed edges
    pos = {i: (X[i, 0], X[i, 1]) for i in range(len(X))}
    nx.draw_networkx_edges(G, pos, ax=ax, arrowstyle='-|>', arrowsize=35, edge_color='black', width=2)
    
    # Plot the nodes with Seaborn style
    nx.draw_networkx_nodes(G, pos, nodelist=range(a), node_color=sns.color_palette("Blues", n_colors=1), node_shape='o', node_size=1450)
    nx.draw_networkx_nodes(G, pos, nodelist=range(a, len(X)), node_color=sns.color_palette("Blues", n_colors=1), node_shape='o', node_size=1450)
    
    # Add labels to the nodes
    labels = {i: f'X{i}' for i in range(len(X))}
    nx.draw_networkx_labels(G, pos, labels, font_size=12, font_color='white')

    ax.set_title("Bayesian Network", fontsize=16)
    plt.axis('off')  # Remove the axis
    plt.tight_layout()
    plt.show()

from itertools import combinations

def plot_factor_graph_cliques(X: np.ndarray, network: np.ndarray):
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    # Create a bipartite graph
    G = nx.Graph()
    variable_nodes = list(range(len(X)))
    factor_nodes = []
    pos = {i: (X[i, 0], X[i, 1]) for i in range(len(X))}

    # Add variable nodes
    G.add_nodes_from(variable_nodes, bipartite=0)
    
    # Identify cliques and create factor nodes
    factor_index = len(X)
    cliques = list(nx.find_cliques(nx.Graph(network)))
    for clique in cliques:
        if len(clique) > 1:
            factor_node = factor_index
            factor_index += 1
            factor_nodes.append(factor_node)
            for var in clique:
                G.add_edge(var, factor_node)
                # Position factor nodes between related variables
                pos[factor_node] = np.mean(X[clique], axis=0)

    G.add_nodes_from(factor_nodes, bipartite=1)

    # Plot the graph
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='black', width=2)
    
    # Plot the variable nodes
    nx.draw_networkx_nodes(G, pos, nodelist=variable_nodes, node_color=sns.color_palette("Blues", n_colors=1), node_shape='o', node_size=1200, label='Variable Nodes')
    # Plot the factor nodes
    nx.draw_networkx_nodes(G, pos, nodelist=factor_nodes, node_color=sns.color_palette("Reds", n_colors=1), node_shape='s', node_size=1200, label='Factor Nodes')
    
    # Add labels to the variable nodes
    variable_labels = {i: f'X{i}' for i in range(len(X))}
    factor_labels = {factor_nodes[i]: f'F{i}' for i in range(len(factor_nodes))}
    labels = {**variable_labels, **factor_labels}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_color='black')

    ax.set_title("Factor Graph with Cliques", fontsize=16)
    plt.axis('off')  # Remove the axis
    plt.tight_layout()
    plt.show()


def plot_display_KDE(n_particles=3):
    # Generate sample data from multiple Gaussian distributions
    n_particles = 10
    np.random.seed(42)
    data1 = np.random.normal(loc=0.8, scale=1.3, size=500)
    data2 = np.random.normal(loc=3.2, scale=2.1, size=500)
    data3 = np.random.normal(loc=-4.1, scale=1.7, size=500)

    # Combine the data into one dataset
    data = np.concatenate([data1, data2, data3])

    # Create the KDE
    kde = gaussian_kde(data, bw_method='silverman')

    # Create a range of values over which to evaluate the KDE
    x = np.linspace(-10, 10, 1000)
    kde_values = kde(x)

    # Plot the histogram of the data
    plt.figure(figsize=(10, 4))
    #plt.hist(data, bins=50, density=True, alpha=0.5, label='Histogram of data')

    # Plot the KDE as a smooth line
    plt.plot(x, kde_values, label='True PDF', color='red')

    # Create individual Gaussians for plotting
    def gaussian(x, mean, std):
        return (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

    # Resample particles from the KDE
    particles = kde.resample(n_particles).flatten()

    # Plot each Gaussian at the resampled particle locations
    for i, particle in enumerate(particles):
        y = gaussian(x, particle, np.std(data)*0.5)
        y /= np.trapz(y, x)  # Normalize each Gaussian to have an area of 1
        plt.plot(x, y*0.2, linestyle='--')

    # Create KDE of the resampled particles
    approx_kde = gaussian_kde(particles, bw_method=0.3, weights=kde(particles.T))
    approx_kde_values = approx_kde(x)
    plt.plot(x, approx_kde_values, label='Approximation', color='green')

    plt.title(f'Effect of KDE on Gaussian-like Signals with {n_particles} Particles')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_2d_KDE():
    # Generate sample data from a 2D Gaussian distribution
    np.random.seed(42)
    mean1 = [0, 0]
    cov1 = [[1, 0.5], [0.5, 1]]
    data1 = np.random.multivariate_normal(mean1, cov1, 300)
    
    mean2 = [3, 3]
    cov2 = [[1, -0.5], [-0.5, 1]]
    data2 = np.random.multivariate_normal(mean2, cov2, 300)
    
    # Combine the data into one dataset
    data = np.vstack([data1, data2])
    
    # Create the KDE
    kde = gaussian_kde(data.T, bw_method='scott')
    
    # Create a grid of values over which to evaluate the KDE
    x = np.linspace(-5, 8, 100)
    y = np.linspace(-5, 8, 100)
    X, Y = np.meshgrid(x, y)
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kde(positions).T, X.shape)
    
    # Plot the KDE as a contour plot
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, Z, levels=20, cmap='Blues')
    plt.scatter(data[:, 0], data[:, 1], s=5, color='red', alpha=0.5, label='Sampled Particles')
    plt.title('2D KDE of Gaussian-like Signals')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.show()

def plot_3d_KDE():
    # Generate sample data from multiple 2D Gaussian distributions
    np.random.seed(42)
    mean1 = [0, 0]
    cov1 = [[1, 0.5], [0.5, 1]]
    data1 = np.random.multivariate_normal(mean1, cov1, 300)
    
    mean2 = [3, 3]
    cov2 = [[1, -0.5], [-0.5, 1]]
    data2 = np.random.multivariate_normal(mean2, cov2, 300)
    
    # Combine the data into one dataset
    data = np.vstack([data1, data2])
    
    # Create the KDE
    kde = gaussian_kde(data.T, bw_method='scott')
    
    # Create a grid of values over which to evaluate the KDE
    x = np.linspace(-5, 8, 100)
    y = np.linspace(-5, 8, 100)
    X, Y = np.meshgrid(x, y)
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kde(positions).T, X.shape)
    
    # Plot the KDE as a 3D surface plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='Blues', edgecolor='k')
    ax.scatter(data[:, 0], data[:, 1], np.zeros_like(data[:, 0]), color='red', alpha=0.5, s=5, label='Sampled Particles')
    
    ax.set_title('3D KDE of Gaussian-like Signals')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Density')
    plt.legend()
    plt.show()

def generate_anchors(deployment_area: np.ndarray, anchor_count: int, border_offset: float) -> np.ndarray:
    x_min, x_max, y_min, y_max = deployment_area
    width = x_max - x_min
    height = y_max - y_min

    # Adjust deployment area to include border offset
    x_min += border_offset
    x_max -= border_offset
    y_min += border_offset
    y_max -= border_offset
    width -= 2 * border_offset
    height -= 2 * border_offset

    # Calculate number of points along each axis, adjusted for equal spacing
    points_per_axis = int(np.ceil(np.sqrt(anchor_count)))
    
    # Ensure we have enough points
    if points_per_axis ** 2 < anchor_count:
        points_per_axis += 1

    # Generate anchors using a hexagonal grid pattern
    anchors = []
    x_step = width / (points_per_axis - 1)
    y_step = height / (points_per_axis - 1)
    
    for j in range(points_per_axis):
        for i in range(points_per_axis):
            x = x_min + i * x_step
            y = y_min + j * y_step
            # Offset every other row for hexagonal pattern
            if j % 2 == 0:
                x += x_step / 2
            if len(anchors) < anchor_count:
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    anchors.append([x, y])
    
    anchors = np.array(anchors)
    
    #plt.scatter(anchors[:, 0], anchors[:, 1])
    #plt.show()
    #print(anchors)
    return anchors

def plot_all(iteration, M, weights, X, weighted_means, anchors, intersections, radius, D, B, _rmse, _similarities, uncertainties_dict, overall_uncertainties_list, M_temp, folder):
    fig, axs = plt.subplots(4, 2, figsize=(10, 20))
    axs = axs.flatten()  # Flatten the 2D array of axes into a 1D array for easy indexing
    
    # Ensure the folder exists
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Plot RMSE
    plot_RMSE(_rmse, iteration+1, ax=axs[0])
    fig.savefig(os.path.join(folder, f"rmse_iter_{iteration}.png"))

    # Plot results
    plot_results(M, B, weights, X, weighted_means, anchors, intersections, iteration, show_bbox=False, uncertainties=None, ax=axs[1])
    fig.savefig(os.path.join(folder, f"results_iter_{iteration}.png"))

    # Plot comparison of graphs
    if iteration == 0:
        plot_compare_graphs(X, weighted_means, anchors, radius, D, iteration, M=M_temp, intersections=intersections, axs=axs[2:4])
    else:
        plot_compare_graphs(X, weighted_means, anchors, radius, D, iteration, M=M, intersections=intersections, axs=axs[2:4])
    fig.savefig(os.path.join(folder, f"compare_graphs_iter_{iteration}.png"))

    # Plot Procrustes similarities
    plot_procrustes(_similarities, iteration+1, ax=axs[4])
    fig.savefig(os.path.join(folder, f"procrustes_iter_{iteration}.png"))

    # Plot uncertainties
    plot_uncertainties(uncertainties_dict=uncertainties_dict, overall_uncertainties_list=overall_uncertainties_list, ax=axs[5])
    fig.savefig(os.path.join(folder, f"uncertainties_iter_{iteration}.png"))

    # Plot network error
    plot_network_error(weighted_means, X, anchors, D, B, ax=axs[6])
    fig.savefig(os.path.join(folder, f"network_error_B_iter_{iteration}.png"))

    plot_network_error(weighted_means, X, anchors, D, np.ones_like(D), ax=axs[7])
    fig.savefig(os.path.join(folder, f"network_error_ones_iter_{iteration}.png"))

    """ # Hide the last subplot if not used
    fig.delaxes(axs[7]) """
    
    plt.tight_layout()
    plt.show()



def save_individual_plots(iteration, M, weights, X, weighted_means, anchors, intersections, radius, D, B, _rmse, _similarities, uncertainties_dict, overall_uncertainties_list, M_temp, folder):
    # Ensure the folder exists
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Plot RMSE
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_RMSE(_rmse, iteration+1, ax=ax)
    fig.savefig(os.path.join(folder, f"rmse_iter_{iteration}.png"))

    plt.close(fig)

    # Plot results
    fig, ax = plt.subplots()
    plot_results(M, weights, X, weighted_means, anchors, intersections, iteration, show_bbox=False, uncertainties=None, ax=ax)
    plt.show()
    fig.savefig(os.path.join(folder, f"results_iter_{iteration}.png"))
    plt.close(fig)

    # Plot comparison of graphs
    fig, ax = plt.subplots(1, 2, figsize=(12,6), sharex=True, sharey=True)

    #plot_compare_graphs(X, weighted_means, anchors, radius, D, iteration, M=M, intersections=intersections, axs=ax)
    XX = X.copy()
    n_anchors = anchors.shape[0]
    XX[n_anchors:] = weighted_means

    plot_network(XX, B, n_anchors=anchors.shape[0], r=radius, D=D, subset=-1, ax=ax[1], name="Estimated Graph")
    plot_network(X, B, n_anchors=anchors.shape[0], r=radius, D=D, subset=-1, ax=ax[0], name="True Graph")
    plt.tight_layout()
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.show()
    fig.savefig(os.path.join(folder, f"compare_graphs_iter_{iteration}.png"))
    plt.close(fig)

    # Plot Procrustes similarities
    fig, ax = plt.subplots()
    plot_procrustes(_similarities, iteration+1, ax=ax)
    plt.show()
    fig.savefig(os.path.join(folder, f"procrustes_iter_{iteration}.png"))
    plt.close(fig)

    # Plot uncertainties
    fig, ax = plt.subplots()
    plot_uncertainties(uncertainties_dict=uncertainties_dict, overall_uncertainties_list=overall_uncertainties_list, n_anchors=anchors.shape[0], ax=ax)
    plt.show()
    fig.savefig(os.path.join(folder, f"uncertainties_iter_{iteration}.png"))
    plt.close(fig)

    # Plot network error with B
    fig, ax = plt.subplots()
    plot_network_error(weighted_means, X, anchors, D, B, ax=ax)
    plt.show()
    fig.savefig(os.path.join(folder, f"network_error_B_iter_{iteration}.png"))
    plt.close(fig)

    # Plot network error with ones
    fig, ax = plt.subplots()
    plot_network_error(weighted_means, X, anchors, D, np.ones_like(D), ax=ax)
    plt.show()
    fig.savefig(os.path.join(folder, f"network_error_ones_iter_{iteration}.png"))
    plt.close(fig)

