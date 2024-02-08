import numpy as np
from sklearn.metrics import euclidean_distances
from matplotlib import pyplot as plt
from scipy.stats import norm, gaussian_kde, uniform
from _COLO import *

def create_bbox(D: np.ndarray, anchors: np.ndarray, limits: np.ndarray):
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
    # Figure out a way to center the target nodes
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

    # Generate particles from bboxes and calculate prior beliefs
    for i in range(n_anchors, n_samples):
        #intersections[i] = np.array([-m/2, m/2, -m/2, m/2])
        bbox = intersections[i].reshape(-1, 2)
        for j in range(d):
            #all_particles[i, :, j] = np.random.uniform(0, m, size=n_particles)
            all_particles[i, :, j] = np.random.uniform(bbox[j, 0], bbox[j, 1], size=n_particles)
        prior_beliefs[i] = mono_potential_bbox(intersections[i])(all_particles[i])

    return all_particles, prior_beliefs


def detection_probability(X_r: np.ndarray, x_u: np.ndarray, radius: int):
    """
    Compute the probability that a node with center `xu` is present at each particle of node t
    Parameters
    ----------
    X_r : array_like, shape (n_particles, 2)
        The coordinates of the particles representing node t
    x_u : array_like, shape (2, )
        The coordinate of the query point u
    radius : int
        Radius of the circular region around u to consider
    Returns
    -------
    probabilities : array_like, shape (n_particles, )
        Probabilities associated with each particle of node t
    """
    # Should radius be max communication range or distance between node r and node u, d_ru?
    dp = np.exp(-np.linalg.norm(X_r - x_u, axis=1)**2 / (2*radius**2))
    dp /= dp.sum()
    return dp

class NBP:
    def __init__(self, X_true: np.ndarray, n_anchors: int, deployment_area: np.ndarray, communication_range: int) -> None:
        self._X_true = X_true
        self._anchors = self._X_true[:n_anchors]
        self._deployment_area = deployment_area
        self._D = get_distance_matrix(X_true, noise=None)
        self._communication_range = communication_range
        self._networks = get_graphs(self._D, communication_range)
        self._n_samples = self._X_true.shape[0]
        self._n_anchors, self._d_dim = self._anchors.shape
        self._anchor_list = list(range(self._n_anchors))

    def iterative_NBP(self, network_type: str, n_particles: int, k: int, n_iter: int=100, communication_range: int=None):
        if communication_range is not None:
            assert communication_range > 0, "Communication range must be greater than zero"
            self._communication_range = communication_range
            self._networks = get_graphs(self.D, communication_range)

        assert network_type in ["full", "one", "two"]

        self.graph = self._networks[network_type]
        self.intersections = create_bbox(D=self.graph, anchors=self._anchors, limits=self._deployment_area)
        self.all_particles, self.prior_beliefs = generate_particles(self.intersections, self._anchors, n_particles) # generate from center
        
        self.messages = np.ones((self._n_samples, self._n_samples, n_particles))
        self.weights = self.prior_beliefs / np.sum(self.prior_beliefs, keepdims=True)
        _rmse = []
        for iter in range(n_iter):
            kde_ru = self.compute_messages(self.messages, self.weights)
            self.compute_beliefs(kde_ru, n_particles=n_particles, k=k)
            guesses = np.einsum('ijk,ij->ik', self.all_particles[self._n_anchors:], self.weights[self._n_anchors:])
            _rmse.append(RMSE(self._X_true[self._n_anchors:], guesses))
            print(f"rmse: {_rmse[-1]}")

            plt.plot(np.arange(iter + 1), _rmse)
            plt.ylabel("RMSE")
            plt.xlabel("iteration")
            plt.show()

            plt.scatter(self._X_true[:, 0], self._X_true[:, 1], c="g", marker="P", label="true", s=50)
            plt.scatter(self._anchors[:, 0], self._anchors[:, 1], c="r", marker="*", label="anchors", s=50)
            plt.scatter(guesses[:, 0], guesses[:, 1], c="y", marker="X", label="predicts", s=50)
            plt.plot([self._X_true[self._n_anchors:, 0], guesses[:, 0]], [self._X_true[self._n_anchors:, 1], guesses[:, 1]], "k--")
        
            for counter in range(self._n_anchors, self._n_samples):
                bbox = self.intersections[counter]
                xmin, xmax, ymin, ymax = bbox
                plt.scatter(self.all_particles[counter, :, 0], self.all_particles[counter, :, 1], s=8)
                plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin])
            plt.legend()
            plt.show()


    def compute_messages(self, messages: np.ndarray, weights: np.ndarray):
        kde_ru = dict()
        initial_guesses = np.einsum('ijk,ij->ik', self.all_particles, weights)

        for sender_r, particles_r in enumerate(self.all_particles): # r sender
            for receiver_u, particles_u in enumerate(self.all_particles): # u receiver
                d_ru = self.graph[sender_r, receiver_u]
                if receiver_u in self._anchor_list or sender_r == receiver_u or d_ru == 0:
                    continue

                if d_ru <= self._communication_range:
                    d_xy = relative_spread(particles_u, particles_r, d_ru)
                    x_ru = particles_r + d_xy
                    detection_prob = detection_probability(x_ru, initial_guesses[receiver_u], self._communication_range)
                    w_ru = detection_prob * (weights[sender_r] / messages[sender_r, receiver_u])
                    w_ru /= w_ru.sum()
                    
                    kde_ru[sender_r, receiver_u] = gaussian_kde(dataset=x_ru.T, weights=w_ru, bw_method='silverman')

                else:
                    kde_ru[sender_r, receiver_u] = lambda: 1 - weights[sender_r] @ detection_probability(particles_r, initial_guesses[receiver_u], radius=self._communication_range).T # not correct
        return kde_ru

    def compute_beliefs(self, kde_ru: dict, n_particles: int, k: int):
        temp_particles = self.all_particles.copy()
        for receiver_u in range(self._n_samples):
            if receiver_u in self._anchor_list:
                continue
            new_particles = draw_particles(self, receiver_u=receiver_u, kde_ru=kde_ru, n_particles=n_particles, k=k)

            q = []
            p = []
            incoming_message = dict()
            for sender_v in range(self._n_samples):
                d_vu = self.graph[receiver_u, sender_v]
                if receiver_u == sender_v or d_vu == 0:
                    continue
                
                if d_vu <= self._communication_range:
                    m_vu = kde_ru[sender_v, receiver_u](new_particles.T)
                    q.append(m_vu)
                else:
                    m_vu = kde_ru[sender_v, receiver_u]()
                p.append(m_vu)
                incoming_message[sender_v] = m_vu
            
            proposal_product = np.prod(p, axis=0)
            proposal_sum = np.sum(q, axis=0)

            W_u = proposal_product / proposal_sum
            W_u /= W_u.sum()

            idxs = np.arange(k*n_particles)
            indicies = np.random.choice(idxs, size=n_particles, replace=True, p=W_u)

            self.all_particles[receiver_u] = new_particles[indicies]
            self.weights[receiver_u] = W_u[indicies]
            self.weights[receiver_u] /= self.weights[receiver_u].sum()
            for neighbour, message in incoming_message.items():
                self.messages[receiver_u, neighbour] = message[indicies]


    def _plot_kde_particle_spread(self, x_ru: np.ndarray, all_particles, node_r: int, node_u: int, d_ru: int):
        """Plot the KDE particle spread from a given node to other node"""
        plt.scatter(self._X_true[node_r, 0], self._X_true[node_r, 1], c='r', label=f"true pos of {node_r}")
        plt.scatter(self._X_true[node_u, 0], self._X_true[node_u, 1], c='r', label=f"true pos of {node_u}")
        plt.scatter(all_particles[node_r, :, 0], all_particles[node_r, :, 1], label=f"particle of sender {node_r}")
        plt.scatter(all_particles[node_u, :, 0], all_particles[node_u, :, 1], label=f"particle of receiver {node_u}")
        plt.scatter(x_ru[:, 0], x_ru[:, 1], label=f"kde of sender {node_r}")
        plt.legend()
        plt.title(f"distance between {node_r} and {node_u} is {d_ru}")
        plt.show()
                    
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
        if iter == 0:
            weighted_means = np.einsum('ijk,ij->ik', M[n_anchors:], prior_beliefs[n_anchors:])
        else:
            weighted_means = np.einsum('ijk,ij->ik', M[n_anchors:], weights[n_anchors:])
        #weighted_means = np.einsum('ijk,ij->ik', M[n_anchors:], weights[n_anchors:])
        #weighted_means = np.einsum('ijk,ij->ik', M[n_anchors:], prior_beliefs[n_anchors:])
        ############################ PLOTING #############################################
        """ plt.scatter(anchors[:, 0], anchors[:, 1], label="anchors", c="r", marker='*') # anchors nodes
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
    

        plt.show() """
        ################################################################################
        m_ru = dict()
        M_new = [[] for i in range(n_samples)]
        
        kn_particles = [k * n_particles for _ in range(n_samples)]
        #neighbour_count = [np.count_nonzero((u[n_anchors:] > 0) & (u[n_anchors:] < radius)) for u in D]
        neighbour_count = [np.count_nonzero((u > 0) & (u < radius)) for u in D]
        print(f"neighbour_count: {neighbour_count}")
        for r, Mr in enumerate(M):
            # sender
            #plt.scatter(Mr[:, 0], Mr[:, 1], label=f"particles of {r}")
            """ if r in anchor_list:
                continue """
            
            for u, Mu in enumerate(M): # skip anchors?
                # receiver
                if u in anchor_list:
                    continue
                if r != u and D[r, u] != 0 and D[r, u] <= radius:
                    
                    if iter % 5 == 0:
                        v = np.random.normal(0, 1, size=n_particles)*0
                        thetas = np.random.uniform(0, 2*np.pi, size=n_particles)
                        cos_u = (D[r, u] + v) * np.cos(thetas)
                        sin_u = (D[r, u] + v) * np.sin(thetas)
                        x_ru = Mr + np.column_stack([cos_u, sin_u])
                    else:

                        d_xy = relative_spread(Mu, Mr, D[r, u])
                        x_ru = Mr + d_xy
                
                    pd = detection_probability(x_ru, weighted_means[u - n_anchors], radius)
                    w_ru = pd * (weights[r] / messages[r, u])
                    w_ru /= w_ru.sum()

                    kde = gaussian_kde(x_ru.T, weights=w_ru, bw_method='silverman')
                    m_ru[r, u] = kde
                    # kde constructed with particles of r to evaluate resampled particles of u

                    new_n_particles = kn_particles[u] // neighbour_count[u]
                    kn_particles[u] -= new_n_particles
                    neighbour_count[u] -= 1

                    sampled_particles = kde.resample(new_n_particles).T
                    M_new[u].append(sampled_particles)
                    #print(anchor_list, u, r)
                    """ if u not in anchor_list and r not in anchor_list:
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
                        plt.show() """
                   
        for qq, Mu_new in enumerate(M_new):
            if len(Mu_new) != 0:
                M_new[qq] = np.concatenate(Mu_new)

        M_temp = M.copy()
        weights_temp = weights.copy()
        for u, Mu_new in enumerate(M_new): # skip anchors?
            # receiver
            if u in anchor_list:
                continue

            incoming_message = dict()
            q = []
            p = []
            for v, Mv in enumerate(M_temp): # if has no neighbour? keep this M or use something else or dont update at ln381
                # sender
                if D[u, v] != 0:
                    if D[u, v] < radius:
                        """ if v in anchor_list:
                            pd = np.array([np.sum(weights_temp[v] * detection_probability(Mv, xu, radius)) for xu in Mu_new])
                            m_vu = pd * duo_potential(Mu_new, Mv[0], D[v, u], np.var(Mu_new))
                            
                        else: """
                        m_vu = m_ru[v, u](Mu_new.T) # Mu_new is the sampled particles from u's neighbour kde's
                        """ pd = np.array([np.sum(weights_temp[v] * detection_probability(Mv, xu, radius)) for xu in Mu_new])
                        m_vu = pd * duo_potential(Mu_new, Mv[0], D[v, u], np.var(Mu_new)) """
                            
                        #m_vu *= np.array([np.sum(weights_temp[v] * detection_probability(Mv, xu, radius)) for xu in Mu_new]) 
                        # m_ru[v, u] is the kde constructed with particles of v (r == v) with distance to u
                        q.append(m_vu)
                    else:
                        pd = np.array([1 - np.sum(weights_temp[v] * detection_probability(Mv, xu, radius)) for xu in Mu_new])
                        m_vu = pd
                    # m_vu is messages of particles from v to u
                    
                    p.append(m_vu)
                    incoming_message[v] = m_vu
                    #new_messages[u, v] = m_vu # message that u receives from v
            
            proposal_product = np.prod(p, axis=0)
            proposal_sum = np.sum(q, axis=0)
            
            W_u = proposal_product / proposal_sum
            #W_u *= mono_potential_bbox(intersections[u])(Mu_new)
            W_u /= W_u.sum()
        
            idxs = np.arange(k*n_particles)
            try:
                indices = np.random.choice(idxs, size=n_particles, replace=True, p=W_u)
            except:
                print(f"iteration: {iter}, u: {u}")
                bbox = intersections[u]
                xmin, xmax, ymin, ymax = bbox
                plt.scatter(X[u, 0], X[u, 1], c='r', s=9, label=f"true {u} pos")
                plt.scatter(Mu_new[:, 0], Mu_new[:, 1], s=9, label=f"particles of {u}")
                plt.scatter(Mu_new[indices, 0], Mu_new[indices, 1], s=9, label=f"choosen particles of M{u}_new")
                plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin])
                plt.title(f"Particles of {u}")
                plt.show()
                return

            M[u] = Mu_new[indices]
            weights[u] = W_u[indices] #???
            weights[u] /= weights[u].sum()

            """ bbox = intersections[u]
            xmin, xmax, ymin, ymax = bbox
            plt.scatter(M[u, :, 0], M[u, :, 1], s=9, label=f"particles of {u}")
            plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin])
            plt.title(f"Sellected particles of {u}")
            plt.show()
 """
            for neighbour, message in incoming_message.items():
                messages[u, neighbour] = message[indices]
            #messages[u] = new_messages[u, :, indices].T

        weighted_means = np.einsum('ijk,ij->ik', M[n_anchors:], weights[n_anchors:])
        _rmse.append(RMSE(X[n_anchors:], weighted_means))
        print(f"rmse: {_rmse[-1]}")
        """ for u, target  in enumerate(M):
            if u in anchor_list:
                continue
            k_kde = gaussian_kde(M[u - n_anchors].T, bw_method='silverman', weights=weights[u])
            w_weights = k_kde(M_temp[u - n_anchors].T)
            print(M_temp[u -n_anchors].shape, w_weights.shape)
            weighted_means[u] = np.average(M_temp[u], weights=w_weights, axis=0)
        print(weighted_means) """
        if (iter + 1) % 5 == 0:
            #print(D[n_anchors:, n_anchors:])
            #print(euclidean_distances(weighted_means))
            plt.plot(np.arange(iter + 1),  _rmse)
            plt.ylabel("RMSE")
            plt.xlabel("iteration")
            plt.show()
            plt.scatter(anchors[:, 0], anchors[:, 1], label="anchors", c="r", marker='*') # anchors nodes
            plt.scatter(X[n_anchors:, 0], X[n_anchors:, 1], label="true", c="g", marker='P') # target nodes
            plt.scatter(weighted_means[:, 0], weighted_means[:, 1], label="preds", c="y", marker="X") # preds
            plt.plot([X[n_anchors:, 0], weighted_means[:, 0]], [X[n_anchors:, 1], weighted_means[:, 1]], "k--",)
            

            for counter in range(n_anchors, n_samples):
                bbox = intersections[counter]
                xmin, xmax, ymin, ymax = bbox
                plt.scatter(M[counter, :, 0], M[counter, :, 1], s=8)
                plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin])

            plt.legend()
            plt.show()
        #weights = weights_temp

seed = 17
np.random.seed(seed)
n, d = 30, 2
m = 100
p = 100
r = 25
a = 4
i = 100
k = 5

X_true, area = generate_targets(seed=seed,
                          shape=(n, d),
                          deployment_area=m,
                          n_anchors=a,
                          show=False)

""" X_true = np.array([
    [m/5, m*1/3],
    [m/5-10, m*2/3-10],
    [m*4/5, m*1/3+15],
    [m*4/5, m*2/3],
    [m/2-10, m/2+10],
    [m*2/5-10, m*2/5],
    [m*2/4-2, m*2/4],
    [m*3/5, m*3/5]
])
area = np.array([0, m, 0, m]) """
D = get_distance_matrix(X_true, noise=None)
graphs = get_graphs(D, communication_range=r)
plot_networks(X_true, a, graphs)
network = graphs['two']

iterative_NBP(D=network, X=X_true, anchors=X_true[:a], deployment_area=area, n_particles=p, n_iter=i, k=k, radius=r)

#nbp = NBP(X_true=X_true, n_anchors=a, deployment_area=area, communication_range=r)
#nbp.iterative_NBP(network_type="one", n_particles=p, k=k, n_iter=i)