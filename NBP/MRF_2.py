import numpy as np
from sklearn.metrics import euclidean_distances
from matplotlib import pyplot as plt
import pprint
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

    likelihood = norm.pdf(dru - euclidean_distance)
    return likelihood

from scipy.stats import gaussian_kde
import numpy as np

def one_step_NBP(D: np.array, anchors: np.ndarray, n_particles: int, X:np.ndarray, m: int):
    """
    D: Measured distances between all sensors (nodes)
    anchors: anchors whose locations are known
    n_particles: number of particles for each node
    X: true locations of nodes (n_samples, 2) shaped (first n_anchors are anchors)
    m: determines the deployment area. Deployment area will be (x_min = 0, x_max = m, y_min = 0, y_max = m)
    """

    n_samples = D.shape[0] # number of nodes to localize
    n_anchors = anchors.shape[0] # number of anchors
    anchor_list = list(range(n_anchors)) # first n nodes are the anchors
    target_neighbours = n_samples - n_anchors - 1
    n_targets = n_samples - n_anchors
    k = 1
    new_n_particles = round(k * n_particles / target_neighbours)
    n_particles = (new_n_particles * target_neighbours) // k


    M, prior_beliefs, intersections = generate_particles(D, anchors, n_particles, m)
    # M: n_particles amount of particles for each nodes from their bounded boxes according to anchors
    # prior_beliefs: prior beliefs for each node. shaped (n_samples)
    # intersections: bounded box for each node (n_samples - n_anchors, 4) shaped = each has[x_min, x_max, y_min, y_max]

    beliefs = np.repeat(prior_beliefs, n_particles).reshape(n_samples, n_particles) # initial beliefs (n_samples, n_particles)
    messages = np.ones((n_samples, n_samples, n_particles)) # messages incoming to node r from node u's nth particle
    new_messages = messages.copy()
    # for r, u; r != u; both r and u represents indexes for nodes, k represents index for particle index of node r
    # messages[r, u, k] represents message node r's k particle receives from node u's all particles
    weights = beliefs / np.sum(beliefs, axis=1, keepdims=True) # weights for each nodes, particles
    # initially, all weights of particles for a node are equal and their sum is 1

    for i in range(2):
        Xu_new = np.ones((n_samples, n_samples, new_n_particles, 2)) # not like this
        m_ru = dict()
        for r, Mr in enumerate(M): # for each node r and its particles Mr
            if r in anchor_list:
                continue
            for u, Mu in enumerate(M): # for each node u and its particles Mu
                if u in anchor_list:
                    continue
                if r != u: # if u is not r
                    v = np.random.normal(0, 1, size=n_particles)*0
                    thetas = np.random.uniform(0, 2*np.pi, size=n_particles)
                    cos_u = (D[r, u] + v) * np.cos(thetas)
                    sin_u = (D[r, u] + v) * np.sin(thetas)
                    x_ru = Mr + np.column_stack([cos_u, sin_u])
                    w_ru = weights[r] / messages[r, u]
                    print(weights[r])
                    w_ru /= w_ru.sum()
                    
                    kde = gaussian_kde(x_ru.T, weights=w_ru, bw_method='silverman')
                    Xu_new[u, r] = kde.resample(new_n_particles).T
                    m_ru[r, u] = kde
        
        mask = np.all(Xu_new[n_anchors:, n_anchors:] == np.ones((new_n_particles, 2)), axis=(2, 3))
        Xu_new = Xu_new[n_anchors:, n_anchors:][~mask]
        Xu_new = Xu_new.reshape(n_targets, k*n_particles, 2)
        Xu_new = np.vstack([np.ones((n_anchors, k*n_particles, 2)), Xu_new])
        for v, Mr, in enumerate(M):
            if v in anchor_list:
                continue
            for u, Mu in enumerate(M):
                if u != v:
                    if u in anchor_list:
                        mru = np.array([pairwise_potential(xr, Mu[0], D[u, v], 1) for xr in Xu_new[v]])
                        new_messages[v, u] = mru
                    else:
                        new_messages[v, u] = m_ru[u, v](Xu_new[v].T)
            
            proposal_product = np.prod(new_messages[v, [i for i in range(n_samples) if i != v]], axis=0)
            proposal_sum = np.sum(new_messages[v, [idx for idx in list(range(n_samples)) if idx not in (anchor_list + [v])]], axis=0)
            W_v = proposal_product / proposal_sum
            W_v /= W_v.sum()
            weights[v] = W_v

        messages = new_messages
    for i, n in enumerate(Xu_new):
        plt.scatter(n[:, 0], n[:, 1], label=f"particle of {i}")

    
    plt.scatter(X[:, 0], X[:, 1], label=f"true targets")
    plt.legend()
    plt.show()



"""
To normalize the sum of Gaussian kernels, you would adjust the amplitude of each kernel so that when they are summed,
the total area under the curve is 1. This process is essential in many applications, such as kernel density estimation
and machine learning algorithms, to maintain the probabilistic interpretation of the data.
"""
""" # For each node r
for r, Mr in enumerate(M):
    if r is an anchor:
        continue

    # Initialize an array to hold the updated weights for node r's particles
    updated_weights = np.zeros_like(weights[r])

    # For each neighbor u of node r
    for u, Mu in enumerate(M):
        if r != u:
            # Approximate the message from node u to node r using node u's particles
            kde = gaussian_kde(Mu.T, weights=weights[u])
            message_ru = kde(Mr.T)

            # Use the message to update the weights for node r's particles
            updated_weights += message_ru

    # Normalize the updated weights
    updated_weights /= np.sum(updated_weights)

    # Resample or shift node r's particles based on the updated weights
    # (This step depends on your specific resampling or shifting method)
    M[r] = resample_particles(Mr, updated_weights)

    # Update the weights array with the new weights
    weights[r] = updated_weights """



def iterative_bp(D: np.ndarray, anchors: np.ndarray, n_particles: int, X:np.ndarray, m: int):
    """
    D: Measured distances between all sensors (nodes)
    anchors: anchors whose locations are known
    n_particles: number of particles for each node
    X: true locations of nodes (n_samples, 2) shaped (first n_anchors are anchors)
    m: determines the deployment area. Deployment area will be (x_min = 0, x_max = m, y_min = 0, y_max = m)
    """
    anchor_list = list(range(len(anchors))) # First n nodes are the anchors
    n_samples = D.shape[0] # number of nodes to localize
    n_anchors = anchors.shape[0] # number of anchors

    M, prior_beliefs, intersections = generate_particles(D, anchors, n_particles, m) # M is random particles for each nodes from their bounded boxes
    intersections = [1] * n_anchors + intersections
    
    messages = np.ones((n_samples, n_samples, n_particles)) # Initial messages
    beliefs = np.repeat(prior_beliefs, n_particles).reshape(n_samples, n_particles) # initial_beliefs
    new_messages = messages.copy()
    new_beliefs = beliefs.copy()
    proposal_products = beliefs.copy()
    proposal_sums = beliefs.copy()
    axs = []
    itr = 0

    initial_weights = beliefs / np.sum(beliefs, axis=1, keepdims=True)
    print(f"initial weights: {initial_weights}")
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for itrs in range(1):
        for r, Mr in enumerate(M):
            if r in anchor_list:
                continue
            for u, Mu in enumerate(M):
                if r != u:
                    if u in anchor_list:
                        pass
                    else:
                        v = np.random.normal(0, 1, size=n_particles)*1
                        thetas = np.random.uniform(0, 2*np.pi, size=n_particles)
                        cos_u = (D[r, u] + v) * np.cos(thetas)
                        sin_u = (D[r, u] + v) * np.sin(thetas)
                        x_ru = Mr + np.column_stack([cos_u, sin_u])
                        #print(x_ru.T.shape)
                        plt.scatter(x_ru[:, 0], x_ru[:, 1], c=colors[u], label=f"{r}, {u}")
                        #print((Mr - x_ru).T)
                        #print(gaussian_kde((Mr - x_ru).T, weights=initial_weights[r]).dataset.T)
                        print(x_ru.T.shape, Mr.T.shape)
                        kde = gaussian_kde(Mu.T, bw_method='silverman', weights=initial_weights[u])
                        m_ru = kde(Mu.T)
                        print(f"Message: {m_ru}")
                    
            """ plt.scatter(Mr[:, 0], Mr[:, 1], c=colors[r], marker='^', label=f"particles of {r}")
            plt.scatter(X[r, 0], X[r, 1], c="black", marker='p', label=f"True pos: {r}")
            plt.legend()
            plt.show() """

    for id in range(itr): # Each iteration
        """ for r, Mr in enumerate(M): # for each node r and its particles Mr
            if r in anchor_list: # if node r is anchor no need to calculate its belief, so continue next iteration
                continue
            for p1, xr in enumerate(Mr): # for each particle xr of node r
                for u, Mu, in enumerate(M): # for each node u and its particles Mu
                    if r != u: # calculate message from node u to each particle of node r, xr if r != u
                        new_messages[r, u, p1] = calculate_message_ur(xr, Mu, D, r, u, messages, beliefs, anchor_list)
                
                proposal_products[r, p1] = np.prod(new_messages[r, :, p1]) # according to numerator of equation (14)
                proposal_sums[r, p1] = np.sum(new_messages[r, [n for n in range(n_samples) if n not in anchor_list and n != r], p1]) # according to equation (15)
                new_beliefs[r, p1] =  single_potential_bbox(intersections[r])(xr) * proposal_products[r, p1]  # calculating beliefs for each particle """



        target_particles = M[n_anchors:]
        target_beliefs = new_beliefs[n_anchors:]

        target_proposal_prod = proposal_products[n_anchors:]
        target_proposal_sums = proposal_sums[n_anchors:]

        #proposal_weights = target_proposal_prod /  np.sum(target_proposal_prod, axis=1, keepdims=True)
        print(target_beliefs)
        proposal_weights = target_proposal_prod / target_proposal_sums
        proposal_weights = proposal_weights / np.sum(proposal_weights, axis=1, keepdims=True)
        weights = target_beliefs / np.sum(target_beliefs, axis=1, keepdims=True)

        weighted_means = np.einsum('ijk,ij->ik', target_particles, weights)
        true_pos = X[n_anchors:]

        """ resampled_particles = np.empty_like(target_particles)
        print(target_particles.shape, proposal_weights.shape)
        for i in range(len(target_particles)):
            flattened_particles = target_particles[i].reshape(-1, 2)
            sampled_indicies = np.random.choice(np.arange(len(flattened_particles)), size=n_particles, p=weights[i])
            resampled_particles[i] = flattened_particles[sampled_indicies].reshape(n_particles, 2)
        print(resampled_particles)
        print(target_particles)
        print(weights)
        print(proposal_weights)

        for i, target in enumerate(target_particles):
            plt.scatter(target[:, 0], target[:, 1], label=f"particles for {i}")

        plt.scatter(anchors[:, 0], anchors[:, 1], c='black', label=f"Anchors")
        plt.scatter(true_pos[:, 0], true_pos[:, 1], c="green", label=f"True Positions", )
        plt.scatter(weighted_means[:, 0], weighted_means[:, 1], c='yellow', label=f"Predictions")
        for i in range(len(true_pos)):
            plt.annotate(i, true_pos[i])
            plt.annotate(i, weighted_means[i])
        plt.legend() """
        

        #print(new_beliefs)
        print(rmse(true_pos, weighted_means))
        beliefs = new_beliefs
        messages = new_messages

        #print(beliefs)
        
        """ print(D)
        # Generate random angles in radians
        degrees = np.random.uniform(0, 2 * np.pi, size=(weighted_means.shape[0], n_particles))

        # Compute sine and cosine of the angles
        sine_values = np.sin(degrees)
        cosine_values = np.cos(degrees)

        # Repeat the weighted_means array along the second axis
        new_particles = np.repeat(weighted_means[:, np.newaxis, :], repeats=n_particles, axis=1)
        # Perform element-wise multiplication for each sample independently
        distances = np.triu(D[n_anchors:, n_anchors:], k=1).ravel()
        distances = distances[distances != 0] / 2
        new_particles = new_particles + distances[:, np.newaxis, np.newaxis] * np.stack([sine_values, cosine_values], axis=2)

        # Plotting
        plt.scatter(weighted_means[:, 0], weighted_means[:, 1], label='predictions')
        plt.scatter(X[n_anchors:, 0], X[n_anchors:, 1], label="origials")

        for particle in new_particles:
            plt.scatter(particle[:, 0], particle[:, 1])

        plt.legend()
        plt.show() """
        #ax = plt.subplot(itr, 1, id+1)
        #plot_results(true_pos, weighted_means, ax, n_anchors, alg="BP", show_lines=True)
        #plt.show()
        

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

import time
#np.random.seed(21)
n_samples, d_dimensions = 12, 2
m_meters = 30
n_particles = 20
r_radius = 20
n_anchors = 4
X = generate_X_points(m_meters, n_samples, d_dimensions)

noise = np.random.normal(0, 2, (n_samples, n_samples))
noise -= np.diag(noise.diagonal())
symetric_noise = (noise + noise.T) / 2
D = euclidean_distances(X) + symetric_noise * 0
one_step_NBP(D, X[:n_anchors], n_particles, X, m_meters)

""" # Define the points and the target point t
area_m =20
n_particles = 3
n_targets = 30
d = 2
resample_c = 300
x_min, x_max, y_min, y_max = 0, area_m, 0, area_m """

""" p1 = np.array([x_min, y_min])
p2 = np.array([x_max, y_min])
p3 = np.array([x_min, y_max])
p4 = np.array([x_max, y_max])
targets = np.array([[0.1, 0.1], [1.1, 1.3], [3.4, 0.88], [1.88, 2.11], [3,3]])
particles = np.array([p1, p2, p3, p4])
weights=np.array([0.25, 0.25, 0.25, 0.25]) """

""" t = np.random.uniform(x_min, x_max, size=d)
rx, ry = (x_max + x_min)/2, (y_min + y_max)/2
o = np.array([rx, ry])


# Messages from Mr to Mu = m_ru
# m_ru = m^(i)_ru(Xu)?
particles = np.repeat(o, n_particles).reshape(n_particles, d) 
#particles = np.random.uniform(rx - (rx/2), ry + (ry/2), size=(n_particles, d)) # Mr
xs = np.random.uniform(t[0] - (rx/2), t[0] + (rx/2), size=n_targets) # Mu_xs
ys = np.random.uniform(t[1] - (ry/2), t[1] + (ry/2), size=n_targets) # Mu_ys
targets = np.column_stack([xs, ys]) # Mu

#targets = t
#targets = targets.reshape(1, -1)
#targets = np.random.uniform(x_min, x_max, size=(n_targets, d))
#print(f"shape: {targets.shape}, t:{t}, targets:{targets}")

d_ru = np.linalg.norm(o - t)
v = np.random.normal(0, 2, size=n_particles)*0
#thetas = np.linspace(0, 2*np.pi, n_particles)
thetas = np.random.uniform(0, 2*np.pi, size=n_particles)
cos_u = (d_ru + v) * np.cos(thetas)
sin_u = (d_ru + v) * np.sin(thetas)
spread_particles = particles + np.column_stack([cos_u, sin_u]) # x_ru

weights = np.ones(n_particles)# Wr weights
weights = np.random.uniform(0, 1, size=n_particles)
weights /= weights.sum()
#weights /= 1 # w_ru Wr / m^(i-1)_ur(Xr == Mr), because init messages are 1

# Perform KDE with the 'silverman' bandwidth method
# Constructing kde from Mr particles for Mu particles
kde = gaussian_kde(spread_particles.T, bw_method='silverman', weights=weights)

# Evaluate the KDE for Mu particles
m = kde(targets.T) # messages for Mu particles? from kde of Mr, x_ru
print(f"test:{kde.pdf(targets.T)}")
resamples_particles = kde.resample(resample_c).T
# Calculate the pairwise potentials considering the measured distance

pp = np.array([pairwise_potential(xr=xr, xu=xu, dru=d_ru, sigma=np.std(spread_particles)*kde.factor) for r, xr in enumerate(targets) for xu in spread_particles])
pp = pp.reshape(targets.shape[0], spread_particles.shape[0])
# Normalize the sum of the pairwise potentials by the prior belief
prior_belief = 1/(area_m +1)
pp *= weights
pp *= prior_belief
pp *= 2
sum_of_particles = np.sum(pp, axis=1)
# Output the results
print(f"KDE result: {m}")
print(f"Manual result: {sum_of_particles}")
error = m - sum_of_particles
print(f"Delta: {np.sqrt(np.mean((m - sum_of_particles)**2))}")
print(np.sum(error)**2)
 """


""" 
#print(n_particles**(1/3)*np.cov(spread_particles))
print(f"kde.cho_cov:{kde.cho_cov}")
print(f"kde.covariance:{kde.covariance}")
print(f"kde.d:{kde.d}")
print(f"kde.factor:{kde.factor}")
print(f"kde.inv_cov{kde.inv_cov}")
print(f"var particles:{np.var(spread_particles)}")
print(f"std particles:{np.std(spread_particles)}")
print(f"scott factor: {n_particles**(-1/(d+4))}")
print(f"neff:{np.sum(weights)**2 / np.sum(weights**2)}")
print(f"silvermann factor: {(n_particles * (d + 2) / 4.)**(-1. / (d + 4))}")
#print(pairwise_potential(o, t, 0, sigma=kde.factor))
dataset = np.array([[0, 4], [4, 0], [3, 3]])
print(dataset)
kde = gaussian_kde(dataset.T)
eva = kde(np.array([[4, 1]]).T)
print(f"new kde.cho_cov:{kde.cho_cov}")
print((np.power(2 * np.pi, -2/2)) * (2 * np.pi))

chol1 = kde.cho_cov[0, 0]
chol2 = kde.cho_cov[1, 1]
normi = np.power(2 * np.pi, -d/2)
print(normi)
normi /= chol1
normi /= chol2

s1 = np.exp(-((4 - 0)**2 + (1 - 4)**2)/2) * normi * (1/3)
s2 = np.exp(-((4 - 4)**2 + (1 - 0)**2)/2) * normi * (1/3)
s3 = np.exp(-((4 - 3)**2 + (1 - 3)**2)/2) * normi * (1/3)

print(s1+s2+s3)
print(pairwise_potential(np.array([0,0]), np.array([3, 4]), 5, 1))
print(eva)
 """

""" plt.scatter(o[0], o[1], label="center")
plt.scatter(t[0], t[1], label="target")
plt.scatter(spread_particles[:, 0], spread_particles[:, 1], label="particles")
plt.scatter(targets[:, 0], targets[:, 1], label="target particles")
plt.scatter(resamples_particles[:, 0], resamples_particles[:, 1], label="resamples")
plt.legend()
#plt.xlim((x_min, x_max))
#plt.ylim((y_min, y_max))
for i, particle in enumerate(spread_particles):
    plt.annotate(str(i), particle)
plt.show() """

""" Z = np.reshape(m, X.shape)
plt.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[x_min, x_max, y_min, y_max])
plt.scatter(test_points[:, 0], test_points[:, 1])
plt.scatter(test_points[np.argmax(m), 0], test_points[np.argmax(m),  1])
plt.show() """
""" #iterative_bp(D, X[:n_anchors], n_particles, X, m_meters)
def measure(n):
    #"Measurement model, return two coupled measurements."
    m1 = np.random.normal(size=n)
    m2 = np.random.normal(size=n)
    return m1+m2, m1-m2
m1, m2 = measure(2000)
xmin = m1.min()
xmax = m1.max()
ymin = m2.min()
ymax = m2.max()
#Perform a kernel density estimate on the data:

X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([m1, m2])

kernel =gaussian_kde(values[:, :20])
plt.scatter(values.T[:20, 0], values.T[:20, 1], label="values")
Z = np.reshape(kernel(positions).T, X.shape)
print(Z.max())
#Plot the results:
print(
    f"pos: {positions.shape}\nZ: {Z.shape}\nval:{values.shape}"
)

fig, ax = plt.subplots()
ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,extent=[xmin, xmax, ymin, ymax])
ax.plot(m1[:20], m2[20:], 'k.', markersize=3)
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
plt.show()
 """

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
            

""" # ... (rest of your code)

# When calculating the message from u to r
for r, Mr in enumerate(M): # for each node r and its particles Mr
    if r in anchor_list: # If node r is anchor node, we already know its location
        continue
    for u, Mu in enumerate(M): # for each node u and its particles Mu
        if r != u: # if u is not r
            if u in anchor_list: # if node u is one of anchors
                pass
            else:
                # ... (rest of your code for generating x_ru)

                # Initialize KDE with particles from node u
                kde = gaussian_kde(Mu.T, weights=weights[u], bw_method='silverman')
                
                # Calculate the pairwise potentials for each particle of node r
                pairwise_potentials = np.array([pairwise_potential(xr, xu, D[r, u], sigma) for xr in Mr for xu in Mu])
                
                # Normalize the pairwise potentials to ensure they sum to 1
                pairwise_potentials /= np.sum(pairwise_potentials)
                
                # Evaluate KDE at the particles of node r, weighted by the normalized pairwise potentials
                m_ur = kde(Mr.T, weights=pairwise_potentials)
 """