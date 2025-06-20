import numpy as np
from sklearn.metrics import euclidean_distances
from matplotlib import pyplot as plt
from scipy.stats import norm, gaussian_kde, uniform
from _COLO import *
from scipy.spatial import procrustes
import seaborn as sns
import pandas
from MDS import ClassicMDS
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import cProfile

""" folder_small = "large_scale1"
folder_big = "results_big"
folder = folder_small
hop_1 = "1hop"
hop_2 = "2hop"
hop = hop_1
window_big = (-55, 55)
window_small = (5, 85)
window = window_small """

def create_bbox(D: np.ndarray, anchors: np.ndarray, limits: np.ndarray, radius: int, priors: bool = False):

    n_samples = D.shape[0]
    n_anchors, d = anchors.shape
    bboxes = np.zeros((n_samples, n_anchors, 2*d))
    intersection_bboxes = np.zeros((n_samples, 2*d))
    if not priors:
        return intersection_bboxes, bboxes
    # Figure out a way to center the target nodes
    gap = 0
    for i in range(n_samples):
        if i < n_anchors:
            for k in range(d):
                intersection_bboxes[i, 2*k] = max(anchors[i, k] - radius, limits[2*k])
                intersection_bboxes[i, 2*k+1] = min(anchors[i, k] + radius, limits[2*k+1])
            continue

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

            if intersection_bboxes[i, 2*k] > intersection_bboxes[i, 2*k+1]:
                intersection_bboxes[i, 2*k], intersection_bboxes[i, 2*k+1] = intersection_bboxes[i, 2*k+1], intersection_bboxes[i, 2*k]
        
            if intersection_bboxes[i, 2*k+1] - intersection_bboxes[i, 2*k] < gap:
                intersection_bboxes[i, 2*k] -= gap
                intersection_bboxes[i, 2*k+1] += gap
    return intersection_bboxes, bboxes

def generate_particles(intersections: np.ndarray, anchors: np.ndarray, n_particles: int, priors, mm, mds_init):
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

    # Generate particles from bboxes and calculate prior beliefs
    if mds_init is not None:
        mds_particles = np.repeat(mds_init[:, np.newaxis, :], repeats=n_particles, axis=1)
        all_particles = mds_particles
    else:
        for i in range(n_anchors, n_samples):
            if not priors:
                intersections[i] = np.array([-mm/2, mm/2, -mm/2, mm/2])
            #intersections[i] = np.array([0, m, 0, m])
            bbox = intersections[i].reshape(-1, 2)
            for j in range(d):
                #all_particles[i, :, j] = np.random.uniform(0, m, size=n_particles)
                all_particles[i, :, j] = np.random.uniform(bbox[j, 0], bbox[j, 1], size=n_particles)
            prior_beliefs[i] = mono_potential_bbox(intersections[i])(all_particles[i])

    return all_particles, prior_beliefs


def detection_probability(X_r: np.ndarray, x_u: np.ndarray, radius: int, dist):
    #print(X_r, x_u)
    """ if dist is not None:
        dp = np.exp(-(np.abs(dist - np.linalg.norm(X_r - x_u, axis=1))**2 / (2*radius**2)))
    else: """
    dp = np.exp(-(np.linalg.norm(X_r - x_u, axis=1)**2 / (2*radius**2)))
    return dp



def iterative_NBP(D: np.ndarray, B:np.ndarray, X: np.ndarray, anchors: np.ndarray,
                  deployment_area: np.ndarray, n_particles: int, n_iter: int,
                    k: int, radius: int, nn_noise: float, benchmark, priors, mds_init):
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
    anchor_list = list(range(n_anchors)) # first n nodes are the anchors
    
    intersections, bboxes = create_bbox(D, anchors, limits=deployment_area, radius=radius, priors=priors)
    mm = deployment_area[0] * -2
    M, prior_beliefs = generate_particles(intersections, anchors, n_particles, priors, mm, mds_init)
    messages = np.ones((n_samples, n_samples, n_particles))
    weights = prior_beliefs / np.sum(prior_beliefs, axis=1, keepdims=True)

    init_particles = M.copy()

    #plot_priors(X, D, 3, anchors, intersections=intersections, bboxes=bboxes, color="red")
    #plot_rings(X, anchors, D, intersections, radius, 4)
    #plot_gaussian_ring(X, a, D)
    plot_MRF(X, B, n_anchors, D, radius, intersections)
    #ax = plot_detection_model(X, D, 4, anchors, radius, intersections)
    #plot_initial_particles_MRF(X, M, weights, n_anchors, D, radius, intersections)
    """ x_D = D*B
    J = calculate_jacobian(X, x_D)
    FIM = calculate_fim(J, nn_noise**2)
    #FIM = calculate_fim(J, 0.1)
    CRLB = np.linalg.inv(FIM)
    variances = np.diag(CRLB)
    overall_crlb = np.sqrt(np.sum(variances))
    print(f"CRLB: {overall_crlb}")
    reshaped = variances.reshape(-1, 2)
    summed = np.sum(reshaped, axis=1) """

    #print(f"max idx: {summed.argmax()}, {summed.max()}")
    #benchmark = max(overall_crlb, benchmark)

    
    KLs = dict()
    for i in range(n_anchors, D.shape[0]):
        KLs[i] = list()

    _rmse = []
    _similarities = []
    plot_rmse = []
    uncertainties_dict = {i: [] for i in range(M.shape[0])}
    overall_uncertainties_list = []
    overall_uncertainties_dict = {}
    error_dict = {}
    folder = "non-sufficient_connectivity"
    # we have different prior beliefs for each node, however we do not use them
    # anywhere, we normalize weights, because all priors are the same, all weights
    # for all particles of all nodes will be the same.

    for iter in range(n_iter):
        #plot_initial_particles(X, D, M, anchors, intersections)
        """ if iter == 0:
            weighted_means = mds_init[n_anchors:]
        else: """
        weighted_means = np.einsum('ijk,ij->ik', M, weights)
        plot_results_initial(anchors, X, weighted_means, intersections, M)
        m_ru = dict()
        M_new = [[] for i in range(n_samples)]
        
        kn_particles = [k * n_particles for _ in range(n_samples)]
        #neighbour_count = [np.count_nonzero((u[n_anchors:] > 0) & (u[n_anchors:] < radius)) for u in D]
        neighbour_count = [np.count_nonzero(u > 0) for u in B]
        # print(neighbour_count)
        for r, Mr in enumerate(M): # sender
            for u, Mu in enumerate(M): # receiver
                """ if u in anchor_list:
                    continue """
                if r != u and D[r, u] != 0 and B[r, u] == 1:
                    if iter == 0: #iter % 3 != 1:#iter == 0 or iter % 4 == 0:
                        d_xy, w_xy = random_spread(particles_r=Mr, d_ru=D[r, u])
                    else:
                        d_xy, w_xy = relative_spread(Mu, Mr, D[r, u])

                    x_ru = Mr + d_xy
                    
                    detect_prob = detection_probability(x_ru, weighted_means[u], radius, dist=D[r, u])
                    w_ru = detect_prob * (weights[r] / messages[r, u]) * (1/w_xy)
                    w_ru /= w_ru.sum()
                    #plot_message_approx(X, D, M, anchors, intersections, x_ru, w_ru, r, u, radius, u_mean=weighted_means[u - n_anchors])
                    
                    kde = gaussian_kde(x_ru.T, weights=w_ru, bw_method='silverman')
                    m_ru[r, u] = kde
                    #plot_kde_ru(kde, Mu, intersections, u, r, x_ru, w_ru)
                    
                    new_n_particles = kn_particles[u] // neighbour_count[u]
                    kn_particles[u] -= new_n_particles
                    neighbour_count[u] -= 1

                    sampled_particles = kde.resample(new_n_particles).T
                    M_new[u].append(sampled_particles)

                    #plot_sampling_from_kde_ur(anchor_list, u, r, X, x_ru, Mu, Mr, sampled_particles, w_ru)
    
        MM_new = M_new.copy()
        for qq, Mu_new in enumerate(M_new):
            if len(Mu_new) != 0:
                M_new[qq] = np.concatenate(Mu_new)

        M_temp = M.copy()
        weights_temp = weights.copy()
        #plot_all_messages(D, B, radius, m_ru, 4, init_particles, weights, X, n_anchors)
        #plot_belief_update(all_particles=M, messages=m_ru, node_of_interest= 4)
        for u, Mu_new in enumerate(M_new): # skip anchors?
            # receiver
            """ if u in anchor_list:
                continue """

            incoming_message = dict()
            q = []
            p = []
            for v, Mv in enumerate(M_temp):
                # sender
                if D[u, v] != 0:
                    if B[u, v] == 1:
                        m_vu = m_ru[v, u](Mu_new.T) # Mu_new is the sampled particles from u's neighbour kde's
                        q.append(m_vu)
                    else:
                        pd = np.array([1 - np.sum(weights_temp[v] * detection_probability(xu, Mv, radius, None)) for xu in Mu_new])
                        m_vu = pd
                    # m_vu is messages of particles from v to u
                    

                    p.append(m_vu)
                    incoming_message[v] = m_vu
            
            proposal_product = np.prod(p, axis=0)
            proposal_sum = np.sum(q, axis=0)
            W_u = proposal_product / proposal_sum
            """ WW_u = W_u.copy()
            WW_u *= mono_potential_bbox(intersections[u])(Mu_new)
            WW_u /= WW_u.sum() """
            W_u /= W_u.sum()
            
        
            idxs = np.arange(k*n_particles)
            try:
                indices = np.random.choice(idxs, size=n_particles, replace=True, p=W_u)
            except:
                print(f"error:{intersections[u]}")
                plot_exception(W_u, u, intersections, X, Mu_new, indices)
                
                return

            M[u] = Mu_new[indices]
            weights[u] = W_u[indices] #???
            weights[u] /= weights[u].sum()
            """ kde_M = gaussian_kde(M[u].T, bw_method='silverman', weights=weights[u])
            M[u] = kde_M.resample(n_particles).T
            weights[u] = kde_M(M[u].T)
            weights[u] /= weights[u].sum() """

            #plot_sellected_particles_of(intersections, Mu_new, M, u)
            for neighbour, message in incoming_message.items():
                messages[u, neighbour] = message[indices]
            
            
            """ prev_particles = M_temp[u]
            prev_weights = weights_temp[u]
            prev_u_belief = gaussian_kde(dataset=prev_particles.T, weights=prev_weights, bw_method="silverman")
            evals = prev_u_belief(M[u].T)
            KL = np.sum(weights[u] * np.log(weights[u]/evals)) """
            
            """ if iter == 0:
                KLs[u] += [KL]
            elif KLs[u][-1] > KL:
                KLs[u] += KL
            else:
                anchor_list.append(u) """

        
        #plot_message_kde(X, M, D, m_ru, anchors, 7, radius, MM_new, weights)

        weighted_means = np.einsum('ijk,ij->ik', M, weights)
        uncertainties, overall_uncertainty = weighted_covariance(particles=M, weights=weights, nodes=weighted_means, uncertainties_dict=uncertainties_dict, overall_uncertainties_list=overall_uncertainties_list, iteration=iter)
        """ overall_uncertainties_dict, error_dict = weighted_covariance_2(
            particles=M[n_anchors:],
            weights=weights[n_anchors:],
            nodes=weighted_means,
            uncertainties_dict=uncertainties_dict,
            overall_uncertainties_dict=overall_uncertainties_dict,
            iteration=iter,
            indicex=indices_hop,
            X_true=X[n_anchors:],
            error_dict=error_dict
        ) """
        """ rmse = RMSE(X[n_anchors:], weighted_means)
        if iter == 0:
            _rmse.append((rmse, overall_crlb))
        elif np.abs(_rmse[-1][0] - rmse) < 0.2:
            _rmse.append((rmse, overall_crlb))
            plot_RMSE(_rmse, iter+1)
            plot_results(M, weights, X, weighted_means, anchors, intersections, iter, show_bbox=False)
            break
        else:
            _rmse.append((rmse, overall_crlb)) """
        
        _, _, disparity = procrustes(X, weighted_means)
        _similarities.append(1 - disparity)
        rmse, median = RMSE(X, weighted_means)
        _rmse.append((rmse, median, benchmark))
        print(f"iter: {iter+1}, rmse: {rmse}")
        #difference_of_distances(weighted_means, X[n_anchors:], radius, B[n_anchors:, n_anchors:])

        """ if len(plot_rmse) == 0 or _rmse[-1] < plot_rmse[-1]+np.random.normal(0.28, 0.25):
            plot_rmse.append(_rmse[-1]) """
        
        #plot_representation(X, anchors, M, weights, 5)
        #plot_results(M, weights, X, weighted_means, anchors, intersections, iteration=None, show_bbox=False)

        if (iter + 1) % 5 == 0:
            #print(overall_uncertainties_dict)
            #plot_uncertainties2(overall_uncertainties_dict, "Uncertainty")
            #plot_uncertainties2(error_dict, "Erros")
            #plot_particle_filter(X, 7, M, D)
            #plot_uncertainties(uncertainties_dict=uncertainties_dict, overall_uncertainties_list=overall_uncertainties_list, n_anchors=anchors.shape[0], ax=None)
            #plot_proposal_dist(M, weights, X, D)
            #plot_initial_particles(X, D, M, anchors, intersections)
            plot_compare_graphs(X, B, weighted_means, anchors, radius, D, iter, M=M_temp, intersections=intersections)
            """ plot_RMSE(_rmse, iter+1) # 1-ax
            plot_results(M, weights, X, weighted_means, anchors, intersections, iter, show_bbox=False, uncertainties=None) # 1-ax
            if iter == 0:
                plot_compare_graphs(X, weighted_means, anchors, radius, D, iter, M=M_temp, intersections=intersections) # 2-ax
            else:
                plot_compare_graphs(X, weighted_means, anchors, radius, D, iter, M=M, intersections=intersections) # 2-ax
            
            plot_procrustes(_similarities, iter+1)# 1-ax
            plot_uncertainties(uncertainties_dict=uncertainties_dict, overall_uncertainties_list=overall_uncertainties_list)# 1-ax
            plot_network_error(weighted_means, X, anchors, D, B)# 1-ax
            plot_network_error(weighted_means, X, anchors, D, np.ones_like(D))# 1-ax """
            #plot_all(iter, M, weights, X, weighted_means, anchors, intersections, radius, D, B, _rmse, _similarities, uncertainties_dict, overall_uncertainties_list, M_temp, folder)
            save_individual_plots(iter, M, weights, X, weighted_means, anchors, intersections, radius, D, B, _rmse, _similarities, uncertainties_dict, overall_uncertainties_list, M_temp, folder)
            #plot_compare_graphs(X, weighted_means, anchors, radius, D, iter, intersections=intersections)
            #plot_results(M, weights, X, weighted_means, anchors, intersections, iteration=None, show_bbox=False)
            #plot_simple_once(X, anchors, D, weighted_means)
            
                
            #plot_probabilistic_vs_deterministic(X, anchors, weights, weighted_means, M)
            #plot_proposal_distributions(X, 7, M, D, m_ru, MM_new)
                
    #plot_RMSE(_rmse, len(_rmse))
    #plot_RMSE(plot_rmse, len(plot_rmse))
            

    return _rmse


def plot_networks(X_true:np.ndarray, n_anchors: int, graphs: dict):
    fig, axs = plt.subplots(len(graphs), 1,  sharex=True, sharey=True, figsize=(6, 6))
    for (title, graph), ax in zip(graphs.items(), axs):
        ax.scatter(X_true[:n_anchors, 0], X_true[:n_anchors, 1], marker="*", c="r", label="anchors")
        ax.scatter(X_true[n_anchors:, 0], X_true[n_anchors:, 1], marker="+", c="g", label="true")
        nx.draw_networkx_edges(nx.from_numpy_array(graph), pos=X_true, ax=ax, width=0.5)
        ax.legend()
        ax.set_title(title)
    plt.show()



def increase_distances(distance_matrix, indices, increase_by):
    N = distance_matrix.shape[0]
    for i in indices:
        for j in range(N):
            if j not in indices and distance_matrix[i, j] != 0:
                distance_matrix[i, j] += increase_by
                distance_matrix[j, i] += increase_by
    return distance_matrix

def sample_gaussian_particles_around_node(uu, p, r):
    # Standard deviation of the Gaussian distribution
    sigma = r / 3  # Approximately 99.7% of data within radius r (3-sigma rule)
    
    # Generate particles from a Gaussian distribution centered at uu
    particles = np.random.normal(loc=uu, scale=sigma, size=(p, 2))
    
    return particles

def plot_computational_time(seconds_distributed, seconds_centralized, hops):
    plt.figure(figsize=(8, 6))
    plt.plot(hops, seconds_distributed, marker='o', linestyle='-', color='b', label='Distributed')
    plt.plot(hops, seconds_centralized, marker='o', linestyle='-', color='r', label='Centralized')
    plt.xlabel('Number of Hops', fontsize=14)
    plt.ylabel('Time (seconds)', fontsize=14)
    plt.title('Computational Time vs. Number of Hops', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Increase the number of ticks on the y-axis
    #y_ticks = np.linspace(0, 5, 11)  # From 0 to 5 seconds, 11 ticks
    plt.yticks(fontsize=12)
    plt.xticks(hops, fontsize=12)
    
    plt.tight_layout()
    plt.show()

def mds_localization(network: np.ndarray, x_true: np.ndarray, num_anchors: int): 
    mds = ClassicMDS(D=network)
    x_hat = mds.classic_mds()
    #mds.plot_results(X_true, x_hat)
    #x_hat_ab = mds.least_squares_registration(anchors=X_true[:num_anchors].copy(), anchors_hat=x_hat[:num_anchors].copy(), X_hat=x_hat)
    #mds.plot_results(X_true, x_hat_ab)
    rmse, median = RMSE(X_true[num_anchors:], x_hat[num_anchors:])
    print(f"MDS RMSE: {rmse}, MDS Median: {median}")
    #return x_hat_ab
    return x_hat

if "__main__" == __name__:

    seed = 31
    np.random.seed(seed)
    n, d = 100, 2
    m = 100
    p = 100
    r = 22
    a = 4
    i = 10
    k = 4

    priors = False
    nn_noise = 1

    X_true, area = generate_targets(seed=seed,
                            shape=(n, d),
                            deployment_area=m,
                            n_anchors=a,
                            show=False)
    
    #X_true[:a] = generate_anchors(deployment_area=area, anchor_count=a)

    """ X_true = np.array([
        [m/5-5, m*1/3+5],
        [m/5-10, m*2/3-10],
        [m*4/5, m*2/3-5],
        [m*4/5, m*1/3+15],
        [m*2/5-10, m*2/5+5],
        [m/2-10, m/2+10],
        [m*3/5, m*3/5],
        [m*2/4-1, m*2/4], 
    ])
    
    area = np.array([0, m, 0, m]) """

    generated_anchors = generate_anchors(deployment_area=area, anchor_count=a, border_offset=np.sqrt(m)*1)
    a = len(generated_anchors)
    X_true[:a] = generated_anchors
    
    D, B = get_distance_matrix(X_true, a, noise=nn_noise, communication_radius=r) # noise : 1
    DD, BB = get_distance_matrix(X_true, a, noise=None, communication_radius=r)
    
    graphs = get_graphs(D)
    graphs_DD = get_graphs(DD)
    #fig, ax = plt.subplots(1, 2, figsize=(6, 12), sharex=True, sharey=True)
    """ plot_network(X_true=X_true,
                 B=B,
                 n_anchors=a,
                 r=r,
                 D=D,
                 name="True Graph",
                 subset=-1, ax=ax[0]) """

    # network is simulated distance matrix
    # n: n hop connectivity
    # nth_n: assumtion of n hop immidiate neighborhood
    mds_network, _, _ = get_n_hop(X_true, D, 15, r, a, 1)
    network, B, C = get_n_hop(X_true, D, 2, r, a, 1)
    zero_count = np.count_nonzero(mds_network == 0)
    print("Number of zeros:", zero_count)
    #print(mds_network)

    mds_results = mds_localization(network=mds_network, x_true=X_true, num_anchors=a)

    #plot_network(X_true, B, n_anchors=a, r=r, D=network, subset=-1)

    DDD = D*B - DD
    non_zero = DDD[DDD != 0]
    benchmark = np.var(non_zero)
    print(f"benchmark: {benchmark}")

    #seconds_distributed = np.array([1.820, 2.432, 3.331, 3.903])
    #seconds_centralized = np.array([7.908, 25.440, 46.664, 65.454])
    #hops = np.array([1, 2, 3, 4])
    #plot_computational_time(seconds_distributed, seconds_centralized, hops)
    #C -= B

    """ print(np.allclose(graphs['one'], D))
    print(np.allclose(graphs['one'], network))
    print(np.allclose(graphs['two'], network))
    print(np.allclose(graphs['two'], D))
    print(np.allclose(network, D)) """
    profiler = cProfile.Profile()
    profiler.enable()
    iterative_NBP(D=network,
                   B=B,
                     X=X_true,
                       anchors=X_true[:a],
                         deployment_area=area,
                           n_particles=p,
                             n_iter=i, k=k,
                               radius=r,
                                 nn_noise=nn_noise,
                                   benchmark=benchmark,
                                   priors=priors,
                                   mds_init=mds_results)
    profiler.disable()
    profiler.dump_stats(f'profile_data_centralized.prof')

    import pstats
    p = pstats.Stats(f'profile_data_centralized.prof')
    p.sort_stats('cumulative').print_stats(5)
