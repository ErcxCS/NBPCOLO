import numpy as np
from sklearn.metrics import euclidean_distances
from matplotlib import pyplot as plt
from scipy.stats import uniform, norm



def plot_results(X, X_hat, ax , a=0, show_lines=False, show_anchors=False, alg=""):

    ax.scatter(X[:, 0], X[:, 1], label="True X")
    ax.scatter(X_hat[:, 0], X_hat[:, 1], label=alg + " Predicted Points")
    for i in range(len(X)):
        ax.annotate(i, X[i])
        ax.annotate(i, X_hat[i])
        
    plt.legend()
    if show_anchors:
        ax.scatter(X[:a, 0], X[:a, 1], "ro")
        ax.scatter(X_hat[:a, 0], X_hat[:a, 1], "go")

    if show_lines:
        for i in range(len(X)):
            ax.plot((X[i, 0], X_hat[i, 0]), (X[i, 1], X_hat[i, 1]), "y--")

def generate_X_points(m, n, d):
    return np.random.uniform(0, m, (n, d))

def rmse(X, X_hat_ab):
    # X is the true coordinates matrix
    # X_hat_ab is the estimated coordinates matrix

    # Compute the Euclidean distance for each node
    error = np.sqrt(np.sum((X - X_hat_ab)**2, axis=1))
    # Compute the RMSE
    return np.sqrt(np.mean(error**2))
"""
def cliques(X: np.ndarray, D: np.ndarray, radius):
    W = D.copy()
    W[W > radius] = 0
    plt.scatter(X[:, 0], X[:, 1], label="True X")
    for clique in nx.clique.find_cliques(nx.from_numpy_array(W)):
        if (len(clique) > 5):
            for i in range(len(clique)):
                plt.annotate(clique[i], X[clique[i]])
                for j in range(i, len(clique)):
                    plt.plot((X[clique[i], 0], X[clique[j], 0]), (X[clique[i], 1], X[clique[j], 1]))
    plt.show()
    #print(nx.find_cycle(nx.from_numpy_array(W)))

"""
import numpy as np
from scipy.stats import norm


def single_potential_bbox(bounded_box: np.ndarray):
    # Define the range of x and y
    x_min, x_max, y_min, y_max = bounded_box
    x_range = (x_min, x_max)
    y_range = (y_min, y_max)

    # Define the PDF for x and y
    def joint_pdf(r):
        if x_min <= r[0] <= x_max and y_min <= r[1] <= y_max:
            return 1 / ((x_max - x_min) * (y_max - y_min))
        else:
            return 0

    return joint_pdf

def generate_particles(D: np.ndarray, anchors: np.ndarray, n_particles: int, m):
    n_anchors = anchors.shape[0]
    n_samples = D.shape[0] - n_anchors
    all_particles = []
    prior_beliefs = [1 for _ in range(n_anchors)]
    intersections = []
    for intersection in find_intersection(D, anchors, m):
        xmin, xmax, ymin, ymax = intersection
        intersections.append(intersection)
        
        plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], c='red')
        xs = np.random.uniform(xmin, xmax, size=n_particles)
        ys = np.random.uniform(ymin, ymax, size=n_particles)
        particles = np.column_stack([xs, ys])
        prior_beliefs.append(single_potential_bbox(intersection)(particles[0]))

        all_particles.append(particles)

    M = np.array(all_particles)
    anchor_particles = np.repeat(anchors[:, np.newaxis, :], repeats=n_particles, axis=1) # anchors will not have particles, so each particle is node's itself
    M = np.concatenate((anchor_particles, M), axis=0)
    prior_beliefs = np.array(prior_beliefs)
    return M, prior_beliefs, intersections

def find_intersection(D: np.ndarray, anchors: np.ndarray, m: int):
    n_samples = D.shape[0]
    n_anchors = len(anchors)
    intersections = []
    # D[D > R] = 0
    dx_min, dx_max, dy_min, dy_max = 0, m, 0, m
    for i in range(n_anchors, n_samples):
        anchor_BBs = np.zeros((n_anchors, 4)) # (x_min, x_max, y_min, y_max)
        for j in range(n_anchors):
            anchor_x, anchor_y = anchors[j]
            half_length = D[i, j]
            if (half_length < 1e-3):
                minimum = np.inf
                for k in range(n_anchors, n_samples):
                    if k != i:
                        if D[k, i] + D[k, j] < minimum and D[k, i] > 1e-3 and D[k, j] > 1e-3:
                            minimum = D[k, i] + D[k, j]
                if minimum == np.inf:
                    minimum = 20/2 # R !something else
                half_length = minimum


            anchor_BBs[j] = [
            max(anchor_x - half_length, dx_min), #x_min
            min(anchor_x + half_length, dx_max), #x_max
            max(anchor_y - half_length, dy_min), #y_min
            min(anchor_y + half_length, dy_max) #y_max
            ] 
            xmin, xmax, ymin, ymax = anchor_BBs[j]
            #
            #plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin])
        intersection_BB = np.array([
            np.max(anchor_BBs[:, 0]), # x_min
            np.min(anchor_BBs[:, 1]), # x_max
            np.max(anchor_BBs[:, 2]), # y_min
            np.min(anchor_BBs[:, 3]) # y_max
        ])
        # xmax < xmin because of noise, fix it
        intersections.append(intersection_BB)
    return intersections

def single_potential(m_initial: int):
    # Define the range of x and y
    m = m_initial
    x_range = (0, m)
    y_range = (0, m)

    # Define the PDF for x and y
    def joint_pdf(r):
        if x_range[0] <= r[0] <= x_range[1] and y_range[0] <= r[1] <= y_range[1]:
            return 1 / ((x_range[1] - x_range[0]) * (y_range[1] - y_range[0]))
        else:
            return 0

    return joint_pdf

def pairwise_potential(xr, xu, dru, sigma):
    """Pair-wise potential function for nodes r and u.
    xr: a state of node r
    xu: a state of node u
    dru: measured distance between node r and node u
    sigma: sigma of noise model
    """
    # Compute the Euclidean distance between xr and xu
    euclidean_distance = np.linalg.norm(xr - xu)
    # Compute the likelihood of dru given xr, xu, and the noise model

    likelihood = norm.pdf(dru - euclidean_distance, scale=sigma)
    #print(norm.pdf(0))
    #print(f"likelihood of {likelihood} given {xr}, {xu} for distance {dru}, euclidean: {euclidean_distance}, delta: {dru - euclidean_distance}")
    return likelihood * np.sqrt(2 * np.pi)

def iterative_bp(D: np.ndarray, anchors: np.ndarray, n_particles: int, X:np.ndarray, m):
    """
    D: (n_samples, n_samples) shaped array Measured distances between all sensors (nodes)
    anchors: (n_anchors, 2) shaped array anchors whose locations are known
    n_particles: number of particles for each node
    X: (n_samples, 2) shaped array coordinates of each node used to compare true locations with estimates
    m: int, represents the deploment area of nodes. Given m, the deployment area is x_min = 0, x_max = m, y_min = 0, y_max = m
    """
    anchor_list = list(range(len(anchors))) # First n nodes are the anchors
    n_samples = D.shape[0] # number of nodes to localize
    n_anchors = anchors.shape[0] # number of anchors whose locations are known

    M, prior_beliefs, intersections = generate_particles(D, anchors, n_particles, m) # random particles for each nodes from their bounded boxes
    # M: Represents the ditribution for each node. Has the shape (n_samples, n_particles, 2)
    intersections = [1] * n_anchors + intersections

    messages = np.ones((n_samples, n_samples, n_particles)) # Initial messages
    #beliefs = np.array([np.array([single_potential(m)(M[i,j]) for j in range(len(M[i]))]) for i in range(len(M))]) # Initial beliefs for each particle
    beliefs = np.repeat(prior_beliefs, n_particles).reshape(n_samples, n_particles) # initial_beliefs
    new_messages = messages.copy()
    new_beliefs = beliefs.copy()
    proposal_products = beliefs.copy()
    proposal_sums = beliefs.copy()
    axs = []
    itr = 5
    print(new_messages.shape)
    for id in range(itr):
        old_beliefs = np.copy(beliefs)
        for r, Mr in enumerate(M):
            if r in anchor_list:
                continue
            for p1, xr in enumerate(Mr):
                for u, Mu, in enumerate(M):
                    if r != u:
                        new_messages[r, u, p1] = calculate_message_ur(xr, Mu, D, r, u, messages, beliefs, anchor_list)
                
                proposal_products[r, p1] = np.prod(new_messages[r, [n for n in range(n_samples) if n not in anchor_list], p1])
                proposal_sums[r, p1] = np.sum(new_messages[r, [n for n in range(n_samples) if n not in anchor_list and n != r], p1])
                new_beliefs[r, p1] =  single_potential_bbox(intersections[r])(xr) * proposal_products[r, p1] # remove log

        #print(new_messages.shape)
        target_particles = M[n_anchors:]
        target_beliefs = new_beliefs[n_anchors:]

        target_proposal_prod = proposal_products[n_anchors:]
        target_proposal_sums = proposal_sums[n_anchors:]

        #proposal_weights = target_proposal_prod /  np.sum(target_proposal_prod, axis=1, keepdims=True)
        
        proposal_weights = target_proposal_prod / target_proposal_sums
        proposal_weights = proposal_weights / np.sum(proposal_weights, axis=1, keepdims=True)
        weights = target_beliefs / np.sum(target_beliefs, axis=1, keepdims=True)

        print(beliefs)

        weighted_means = np.einsum('ijk,ij->ik', target_particles, weights)
        true_pos = X[n_anchors:]

        for i, target in enumerate(target_particles):
            plt.scatter(target[:, 0], target[:, 1], label=f"particles for {i}")

        plt.scatter(anchors[:, 0], anchors[:, 1], c='black', label=f"Anchors")
        plt.scatter(true_pos[:, 0], true_pos[:, 1], c="green", label=f"True Positions", )
        plt.scatter(weighted_means[:, 0], weighted_means[:, 1], c='yellow', label=f"Predictions")
        for i in range(len(true_pos)):
            plt.annotate(i, true_pos[i])
            plt.annotate(i, weighted_means[i])
        plt.legend()
        
        print(rmse(true_pos, weighted_means))
        beliefs = new_beliefs
        messages = new_messages

        """ # Generate random angles in radians
        degrees = np.random.uniform(0, 2 * np.pi, size=(weighted_means.shape[0], n_particles))

        # Compute sine and cosine of the angles
        sine_values = np.sin(degrees)
        cosine_values = np.cos(degrees)

        # Repeat the weighted_means array along the second axis
        new_particles = np.repeat(weighted_means[:, np.newaxis, :], repeats=n_particles, axis=1)

        print(np.stack([sine_values, cosine_values], axis=2))
        # Perform element-wise multiplication for each sample independently
        asd = np.array([6, 2]).reshape(1, -1)
        print(asd.shape)
        qq = new_particles + np.array([6, 2]).reshape(1, -1) * np.stack([sine_values, cosine_values], axis=2)
        print(qq)
        # Plotting
        plt.scatter(weighted_means[:, 0], weighted_means[:, 1], label='Original')

        for q in qq:
            plt.scatter(q[:, 0], q[:, 1])

        plt.legend()
        plt.show() """
        #ax = plt.subplot(itr, 1, id+1)
        #plot_results(true_pos, weighted_means, ax, n_anchors, alg="BP", show_lines=True)
    plt.show()
        

def calculate_message_ur(xr: np.ndarray, Mu: np.ndarray, D: np.ndarray, r: int, u: int, messages: np.ndarray, beliefs: np.ndarray, anchor_list: list):
    """
    Calculates message to xr state (a particle) of node r, from particles of node u (xu states)
    xr: a single state of node r (a particle) which we are calculating message from node u particles (xu states)
    Mu: Particles (states) of node u
    D: (n_samples, n_samples) measured distance matrix
    r: the node which its particle will receive the message from node u
    u: the node which sends message to node r
    messages: (n_samples, n_partciles) messages from previous iteration. message[r, u, p1] is the message received by node r's p1 particle from node u
    beliefs: (n_samples, n_particles) beliefs of each node's particles
    anchor_list: list of anchor indexes
    """
    message = 0
    for p2, xu in enumerate(Mu):
        likelihood_xu_xr = pairwise_potential(xr, xu, D[r, u], 1)
        if u in anchor_list:
            return likelihood_xu_xr
        message += likelihood_xu_xr * (beliefs[u, p2] / messages[u, r, p2])
    return message

np.random.seed(23)
n_samples, d_dimensions = 6, 2
m_meters = 20
n_particles = 8
r_radius = 50
n_anchors = 3
X = generate_X_points(m_meters, n_samples, d_dimensions)

noise = np.random.normal(0, 2, (n_samples, n_samples))
noise -= np.diag(noise.diagonal())
symetric_noise = (noise + noise.T) / 2
D = euclidean_distances(X) + symetric_noise * 0

iterative_bp(D, X[:n_anchors], n_particles, X, m_meters)
print(pairwise_potential(np.array([0, 0]), np.array([3, 4]), 5, 1))


def belief_propagation(X: np.ndarray, D: np.ndarray, m: int, anchor_list : list):
    """
    X: Current estimate for locations of nodes
    D: Measured distances between sensors
    m: Coverage. Deployment range of sensors
    """
            
    beliefs = []
    for i, r in enumerate(X):
        X_est = X.copy()
        if i in anchor_list:
            Mr = (np.array(X[i]), belief(X, m, D, i, anchor_list))
            print(Mr)
        else:
            num_particles = 100
            particles = np.random.uniform(1, m, (num_particles, 2))
            weights = np.zeros(num_particles)
            for k, xy_r in enumerate(particles):
                X_est[i] = xy_r
                weights[k] = belief(X_est, m, D, i, anchor_list)
            #weights = weights / weights.sum()
            Mr = (particles, weights.mean())
            print(Mr)
        beliefs.append(Mr)
    return np.array(beliefs)


def belief(X: np.ndarray, m: int, D: np.ndarray, i: int, anchor_list: list):
    """
    X: Current estimate for locations of nodes
    D: Measured distances between sensors
    m: Coverage. Deployment range of sensors
    i: index of node r, the sensor we are calculating belief for
    """
    piror_belief = single_potential(m)(X[i])
    if i in anchor_list:
        piror_belief = 1
    product = 1
    for j, u in enumerate(X):
        if i != j:
            product *= update_message(X, D, m, i, j, [], anchor_list, 1) # marginalizes

    posterior_belief = piror_belief * product
    print(f"Belief of r:{i}, {X[i]} = {posterior_belief}")
    return posterior_belief

def update_message(X: np.ndarray, D:np.ndarray, m: int, i: int, j: int,
                    visited: list, anchor_list: list, sigma_noise: int):
    """
    X: Current estimate for locations of nodes
    D: Measured distances between sensors
    m: Coverage. Deployment range of sensors
    i: index of node r, messages incoming to this node r from u
    j: index of node u, messages outgoing from this node u to r
    """
    visited.append(j)

    if j in anchor_list:
        product = 1
        for k, p in enumerate(X):
            if not (k == j or k == i or k in visited):
                product *= update_message(X, D, m, j, k, visited, anchor_list, sigma_noise)
        message = pairwise_potential(X[i], X[j], D[i, j], 1) * product
        return message
    else:
        num_particles = 100
        particles = np.random.uniform(1, m, (num_particles, 2))
        message = np.zeros(num_particles)

        for k, xu in enumerate(particles):
            X_est = X.copy()
            X_est[j] = xu
            product = 1

            state_xu = X_est[j]
            prior_of_xy = single_potential(m)(state_xu)
            likelihood_z_xy = pairwise_potential(X[i], state_xu, D[i, j], 1)

            for l, p in enumerate(X):
                if not (l == j or l == i or l in visited):
                    product *= update_message(X_est, D, m, j, l, visited, anchor_list, sigma_noise)

            result = prior_of_xy * likelihood_z_xy * product
            message[k] = result
        #message /= message.sum()
    
        return message.sum()
            




