import numpy as np
from matplotlib import pyplot as plt
from _COLO import *
import cProfile
from concurrent.futures import ThreadPoolExecutor

def create_bbox(D: np.ndarray, anchors: np.ndarray, limits: np.ndarray, gap: int = 0):
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

            if intersection_bboxes[i, 2*k] > intersection_bboxes[i, 2*k+1]:
                intersection_bboxes[i, 2*k], intersection_bboxes[i, 2*k+1] = intersection_bboxes[i, 2*k+1], intersection_bboxes[i, 2*k]
        
            if intersection_bboxes[i, 2*k+1] - intersection_bboxes[i, 2*k] < gap:
                intersection_bboxes[i, 2*k] -= gap
                intersection_bboxes[i, 2*k+1] += gap
    return intersection_bboxes, bboxes

def generate_particles(intersections: np.ndarray, anchors: np.ndarray, n_particles: int, priors: bool, m: int):
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
        if not priors:
            intersections[i] = np.array([-m/2, m/2, -m/2, m/2])
        #intersections[i] = np.array([0, m, 0, m])
        bbox = intersections[i].reshape(-1, 2)
        for j in range(d):
            #all_particles[i, :, j] = np.random.uniform(0, m, size=n_particles)
            all_particles[i, :, j] = np.random.uniform(bbox[j, 0], bbox[j, 1], size=n_particles)
        prior_beliefs[i] = mono_potential_bbox(intersections[i])(all_particles[i])

    return all_particles, prior_beliefs

class NBP:

    def __init__(self,
                 X_true: np.ndarray = None,
                 deployment_area: np.ndarray = None,
                 seed: int = None,

                 n_nodes: int = 8,
                 d_dim: int = 2,
                 meters: int = 100,
                 n_samples: int = 125,
                 radius: int = 33,
                 n_anchors: int = 4,
                 n_iter: int = 20,
                 n_batches: int = 4,
                 n_hop: int = 2,
                 n_neighbors: int = 1,

                 auto_anchors: bool = False,
                 noise: float = None,
                 priors: bool = True,
                 experiment: int = 0) -> None:
        self.seed = seed
        self.n_nodes = n_nodes
        self.d_dim = d_dim
        self.meters = meters
        self.n_samples = n_samples
        self.radius = radius
        self.n_anchors = n_anchors
        self.n_iter = n_iter
        self.n_batches = n_batches
        self.n_hop = n_hop
        self.n_neighbors = n_neighbors
        self.priors = priors
        self.experiment = experiment

        if X_true is not None and deployment_area is not None:
            self.X_true, self.deployment_area = X_true, deployment_area
        else:
            self.X_true, self.deployment_area = generate_targets(seed=self.seed,
                                                                shape=(self.n_nodes, self.d_dim),
                                                                deployment_area=self.meters,
                                                                n_anchors=self.n_anchors,
                                                                show=False)
        self.anchors = self.X_true[:self.n_anchors]
        if auto_anchors:
            self.anchors = generate_anchors(self.deployment_area, self.n_anchors, np.sqrt(m)*1)
            self.n_anchors = len(self.anchors)
            self.X_true[:self.n_anchors] = self.anchors
        
        self.D, self.C = get_distance_matrix(X_true=self.X_true,
                                             n_anchors=self.n_anchors,
                                             communication_radius=self.radius,
                                             noise=noise)
        
        if noise is not None:
            DD, _ = get_distance_matrix(X_true=self.X_true,
                                             n_anchors=self.n_anchors,
                                             communication_radius=self.radius,
                                             noise=None)
            DDD = self.D - DD
            non_zero = DDD[DDD != 0]
            self.variance = np.var(non_zero)
            print(self.variance)

        # Maybe not take connectivity
        self.D, _, _ = get_n_hop(X=self.X_true,
                                    D=self.D,
                                    n=self.n_hop,
                                    r=self.radius,
                                    n_anchors=self.n_anchors,
                                    nth_n=self.n_neighbors)

    def __init_NBP__(self):
        self.anchor_list = list(range(self.n_anchors))
        intersections, bboxes = create_bbox(D=self.D,
                                            anchors=self.anchors,
                                            limits=self.deployment_area,
                                            gap=0)

        all_particles, prior_beliefs = generate_particles(intersections,
                                                          anchors=self.anchors,
                                                          n_particles=self.n_samples,
                                                          priors=self.priors,
                                                          m=self.meters)
        
        messages = np.ones((self.n_nodes, self.n_nodes, self.n_samples))
        weights = prior_beliefs / np.sum(prior_beliefs, axis=1, keepdims=True)
        return messages, weights, all_particles

    def iterative_NBP(self):
        messages, weights, all_particles = self.__init_NBP__()
        self._rmse = []
        self._median = []
        uncertainties_dict = {i: [] for i in range(all_particles.shape[0] - self.n_anchors)}
        self.overall_uncertainties = []
        self.error_neighborhoods = []
        self.similarities = []
        self.differences = []
        for iter in range(self.n_iter):
            messages, weights, all_particles = self.NBP_iteration(all_messages=messages, all_weights=weights, all_particles=all_particles)
            estimates = np.einsum('ijk,ij->ik', all_particles[self.n_anchors:], weights[self.n_anchors:])

            t_rmse, t_median = RMSE(self.X_true[self.n_anchors:], estimates)
            self._rmse.append(t_rmse)
            self._median.append(t_median)
            _, _ = weighted_covariance(all_particles, weights, estimates, uncertainties_dict, self.overall_uncertainties, iter)
            _, _, disparity = procrustes(self.X_true[self.n_anchors:], estimates)
            self.similarities.append(1- disparity)
            p = error_vs_neighborhood(self.n_anchors, self.D, estimates, self.X_true[self.n_anchors:])
            self.error_neighborhoods.append(p)
            #self.differences.append(difference_of_distances(estimates, self.X_true[self.n_anchors:], self.radius, self.C[self.n_anchors:, self.n_anchors]))
            print(f"Experiment: {self.experiment}, iter: {iter + 1}, RMSE: {self._rmse[-1]}")
        
        results = {
            "rmse": self._rmse,
            "median": self._median,
            "CRLB": self.variance,
            "uncertainties": self.overall_uncertainties,
            "similarities": self.similarities,
            "neighborhoods": self.error_neighborhoods,
            #"differences": self.differences
        }
        return results
    
    def NBP_iteration(self, all_messages: np.ndarray, all_weights: np.ndarray, all_particles: np.ndarray):
            estimates = np.einsum('ijk,ij->ik', all_particles[self.n_anchors:], all_weights[self.n_anchors:])
            messages_ru = dict()
            sampled_particles = [[] for _ in range(self.n_nodes)]
            batches_remaining = np.array([self.n_batches * self.n_samples for _ in range(self.n_nodes)])
            neighbor_count = np.count_nonzero(self.C, axis=1)

            def message_approximation(node_r):
                nonlocal batches_remaining, neighbor_count
                particles_r = all_particles[node_r]
                for node_u in range(self.n_nodes):
                    if node_u in self.anchor_list or node_r == node_u or self.D[node_r, node_u] == 0 or self.C[node_r, node_u] != 1:
                        continue

                    if len(self._rmse) == 0:
                        d_xy, W_xy = random_spread(particles_r=particles_r, d_ru=self.D[node_r, node_u])
                    else:
                        particles_u = all_particles[node_u]
                        d_xy, W_xy = relative_spread(particles_u=particles_u, particles_r=particles_r, d_ru=self.D[node_r, node_u])

                    X_ru = particles_r + d_xy
                    difference_sq = np.sum((X_ru - estimates[node_u - self.n_anchors]) ** 2, axis=1)
                    detection_probabilities = np.exp(-(difference_sq / (2 * self.radius ** 2)))
                    W_ru = detection_probabilities * (all_weights[node_r] / all_messages[node_r, node_u]) * (1/W_xy)
                    W_ru /= W_ru.sum()

                    proposal_ru = gaussian_kde(dataset=X_ru.T, weights=W_ru, bw_method='silverman')
                    messages_ru[node_r, node_u] = proposal_ru

                    n_particles = batches_remaining[node_u] // neighbor_count[node_u]
                    batches_remaining[node_u] -= n_particles
                    neighbor_count[node_u] -= 1

                    particles = proposal_ru.resample(n_particles).T
                    sampled_particles[node_u].append(particles)

            with ThreadPoolExecutor() as executor:
                executor.map(message_approximation, range(self.n_nodes))

            for node in range(self.n_nodes):
                if sampled_particles[node]:
                    sampled_particles[node] = np.concatenate(sampled_particles[node])

            temp_all_particles = all_particles.copy()
            temp_all_weights = all_weights.copy()

            def belief_update(node_u):
                if node_u in self.anchor_list:
                    return
                
                particles_u = sampled_particles[node_u]
                incoming_message_u = dict()
                one_hop_messages = []
                all_messages_u = []
                for node_r in range(self.n_nodes):
                    if self.D[node_u, node_r] != 0:
                        if self.C[node_u, node_r] == 1:
                            message_ru = messages_ru[node_r, node_u](particles_u.T)
                            one_hop_messages.append(message_ru)
                        else:
                            difference_sq = np.sum((particles_u[:, None, :] - temp_all_particles[node_r]) ** 2, axis=2)
                            detection_probabilities = np.exp(-(difference_sq / (2 * self.radius ** 2)))
                            received_message_r = 1 - np.sum(temp_all_weights[node_r] * detection_probabilities, axis=1)
                            message_ru = received_message_r

                        all_messages_u.append(message_ru)
                        incoming_message_u[node_r] = message_ru

                proposal_product = np.prod(all_messages_u, axis=0)
                proposal_sum = np.sum(one_hop_messages, axis=0)

                W_u = proposal_product / proposal_sum
                W_u /= W_u.sum()

                indexes = np.random.choice(np.arange(W_u.size), size=self.n_samples, replace=True, p=W_u)

                all_particles[node_u] = particles_u[indexes]
                all_weights[node_u] = W_u[indexes]
                all_weights[node_u] /= all_weights[node_u].sum()

                for neighbor, message in incoming_message_u.items():
                    all_messages[node_u, neighbor] = message[indexes]
                
            with ThreadPoolExecutor() as executor:
                executor.map(belief_update, range(self.n_nodes))
            
            return all_messages, all_weights, all_particles
    
    def NBP_iteration2(self, all_messages: np.ndarray, all_weights: np.ndarray, all_particles: np.ndarray):
            # Intializing iteration
            estimates = np.einsum('ijk,ij->ik', all_particles[self.n_anchors:], all_weights[self.n_anchors:])
            messages_ru = dict()
            sampled_particles = [[] for i in range(self.n_nodes)]
            batches_remaining = [self.n_batches * self.n_samples for _ in range(self.n_nodes)]
            neighbor_count = [np.count_nonzero(u > 0) for u in self.C]

            # Message approximation and sampling from proposals
            for node_r, particles_r in enumerate(all_particles): # r is message sender
                for node_u, particles_u in enumerate(all_particles): # u is message receiver
                    if node_u in self.anchor_list: # if receiver is anchor, skip
                        continue

                    # for all one-hop neighbors of r, approximate all outgoing messages
                    if node_r != node_u and self.D[node_r, node_u] != 0 and self.C[node_r, node_u] == 1:
                        noise = np.random.normal(0, 1, size=self.n_samples)*1
                        thetas = np.random.uniform(0, 2*np.pi, size=self.n_samples)
                        cos_u = (self.D[node_r, node_u] + noise) * np.cos(thetas)
                        sin_u = (self.D[node_r, node_u] + noise) * np.sin(thetas)
                        # Particles of message from r to u
                        X_ru = particles_r + np.column_stack([cos_u, sin_u]) 

                        difference_sq = np.linalg.norm(X_ru - estimates[node_u - self.n_anchors], axis=1)**2
                        detection_probabilities = np.exp(-(difference_sq / (2 * self.radius**2)))
                        # Weights of particles of message from r to u
                        W_ru = detection_probabilities * (all_weights[node_r] / all_messages[node_r, node_u])
                        W_ru /= W_ru.sum()

                        # Proposal distribution of r, for u. Also known as message from r to u approximation
                        proposal_ru = gaussian_kde(dataset=X_ru.T, weights=W_ru, bw_method='silverman')
                        messages_ru[node_r, node_u] = proposal_ru

                        # Sampling from proposals of neighbors
                        n_particles = batches_remaining[node_u] // neighbor_count[node_u]
                        batches_remaining[node_u] -= n_particles
                        neighbor_count[node_u] -= 1

                        particles = proposal_ru.resample(n_particles).T
                        sampled_particles[node_u].append(particles)
            
            # Merge sampled particles of each node from neighbors
            for node, particles_from_neighbors in enumerate(sampled_particles):
                if len(particles_from_neighbors) != 0:
                    sampled_particles[node] = np.concatenate(particles_from_neighbors)

            temp_all_particles = all_particles.copy()
            temp_all_weights = all_weights.copy()

            # Belief computations and Resampling with replacement
            for node_u, particles_u in enumerate(sampled_particles): # u is message receiver
                if node_u in self.anchor_list: # if u is anchor, skip
                    continue

                incoming_message_u = dict()
                one_hop_messages = []
                all_messages_u = []
                
                for node_r, particles_r in enumerate(temp_all_particles): # r is message sender
                    if self.D[node_u, node_r] != 0: # if connected with one-hop or two-hop
                        if self.C[node_u, node_r] == 1: # if u and r one-hop neighbors
                            message_ru = messages_ru[node_r, node_u](particles_u.T) # evaluate sampled particles
                            one_hop_messages.append(message_ru)
                        else: # if u and r two-hop neighbors
                            two_hop_message = []
                            for particle_of_u in particles_u:
                                difference_sq = np.linalg.norm(particle_of_u - particles_r, axis=1)**2
                                detection_probabilities = np.exp(-(difference_sq / (2 * self.radius**2)))
                                two_hop_message_particle = 1 - np.sum(temp_all_weights[node_r] * detection_probabilities)
                                two_hop_message.append(two_hop_message_particle)
                            received_message_r = np.array(two_hop_message)
                            message_ru = received_message_r

                        all_messages_u.append(message_ru)
                        incoming_message_u[node_r] = message_ru
                
                proposal_product = np.prod(all_messages_u, axis=0)
                proposal_sum = np.sum(one_hop_messages, axis=0)

                W_u = proposal_product / proposal_sum # Weights to resample particles of u
                W_u /= W_u.sum()

                indexes = np.arange(self.n_batches * self.n_samples)
                indexes = np.random.choice(indexes, size=self.n_samples, replace=True, p=W_u)

                all_particles[node_u] = particles_u[indexes]
                all_weights[node_u] = W_u[indexes]
                all_weights[node_u] /= all_weights[node_u].sum()

                for neighbor, message in incoming_message_u.items():
                    all_messages[node_u, neighbor] = message[indexes]

            return all_messages, all_weights, all_particles

def run_experiment(parameters, n_experiments, auto_anchors, n_hop, priors, name):
    experiments = dict()
    for runs in range(n_experiments):
        print(f"Experiment - {runs + 1} for {name}")
        X_true, area = generate_targets(seed=parameters['seed'],
                                        shape=(parameters['n_nodes'], parameters['d_dim']),
                                        deployment_area=parameters['meters'],
                                        n_anchors=parameters['n_anchors'],
                                        show=False)

        nbp = NBP(X_true=X_true,
                  deployment_area=area,
                  seed=parameters['seed'],
                  n_nodes=parameters['n_nodes'],
                  d_dim=parameters['d_dim'],
                  meters=parameters['meters'],
                  n_samples=parameters['n_particles'],
                  radius=parameters['radius'],
                  n_anchors=parameters['n_anchors'],
                  n_iter=parameters['n_iter'],
                  n_batches=parameters['n_batch'],
                  n_hop=n_hop,
                  n_neighbors=parameters['n_neighbors'],
                  auto_anchors=auto_anchors,
                  noise=1,
                  priors=priors,
                  experiment=runs + 1)

        profiler = cProfile.Profile()
        profiler.enable()
        experiments[runs] = nbp.iterative_NBP()
        profiler.disable()
        profiler.dump_stats(f'profile_data_{name}.prof')

        import pstats
        p = pstats.Stats(f'profile_data_{name}.prof')
        p.sort_stats('cumulative').print_stats(5)
        
        del nbp

    all_rmse = np.array([exp['rmse'] for exp in experiments.values()])
    all_median = np.array([exp['median'] for exp in experiments.values()])
    all_uncertainties = np.array([exp['uncertainties'] for exp in experiments.values()])
    all_similarities = np.array([exp['similarities'] for exp in experiments.values()])
    all_CRLB = np.array([exp['CRLB'] for exp in experiments.values()])
    #all_differences = np.array([exp['differences'] for exp in experiments.values()])

    avg_rmse = np.mean(all_rmse, axis=0)
    avg_median = np.mean(all_median, axis=0)
    avg_uncertainties = np.mean(all_uncertainties, axis=0)
    avg_similarities = np.mean(all_similarities, axis=0)
    avg_CRLB = np.mean(all_CRLB)
    #avg_differences = np.mean(all_differences, axis=0)

    return avg_rmse, avg_median, avg_uncertainties, avg_similarities, avg_CRLB#, avg_differences

if __name__ == "__main__":
    seed = None
    np.random.seed(seed)
    n, d = 125, 2
    m = 150
    p = 125
    r = 33
    a = 7
    i = 5
    k = 4
    n_neighbors = 1

    parameters = {
        "seed": seed,
        "n_nodes": n,
        "d_dim": d,
        "meters": m,
        "n_particles": p,
        "radius": r,
        "n_anchors": a,
        "n_iter": i,
        "n_batch": k,
        "n_neighbors": n_neighbors,
    }

    n_experiments = 15
    experiment_configs = [
        ("1-hop Placement", False, 1, False),
        ("2-hop Placement", False, 2, False),
        ("3-hop Placement", False, 3, False),
        ("4-hop Placement", False, 4, False),
        #("primitive", False, 1, False),
        #("2_hop", False, 2, False),
        #("priors", False, 1, True),
        #("placement", True, 1, False),
        #("placement + 2_hop", True, 2, False),
        #("placement + priors", True, 1, True),
        #("2_hop + priors", False, 2, True),
        #("combined", True, 2, True)
    ]

    comp_rmse, comp_median, comp_uncertainty, comp_similarity, comp_CRLB = [], [], [], [], []

    for name, auto_anchors, n_hop, priors in experiment_configs:
        rmse, median, uncertainty, similarity, CRLB = run_experiment(parameters, n_experiments, auto_anchors, n_hop, priors, name)
        comp_rmse.append(rmse)
        comp_median.append(median)
        comp_uncertainty.append(uncertainty)
        comp_similarity.append(similarity)
        comp_CRLB.append(CRLB)
        #comp_differences.append(differences)

    # Convert to numpy arrays for easier plotting
    rmse = np.array(comp_rmse)
    median = np.array(comp_median)
    uncertainty = np.array(comp_uncertainty)
    similarity = np.array(comp_similarity)
    CRLB = np.array(comp_CRLB)
    #differences = np.array(comp_differences)

    """ fig, ax = plt.subplots(figsize=(6, 6))

    # Plotting results
    sns.set(style="whitegrid")
    colors = ["blue", "yellow", "cyan", "green", "purple", "magenta"]
    for idx, (name, _, _, _) in enumerate(experiment_configs):
        #color = next(plt.gca()._get_lines.prop_cycler)['color']
        ax.plot(np.arange(i), rmse[idx], color=colors[idx], label=name)
        ax.plot(np.arange(i), median[idx], color=colors[idx], linestyle="--")
        #plt.plot(np.arange(i), differences[idx], linestyle="--", color=colors[idx])
    
    ax.plot(np.arange(i), np.repeat(np.mean(CRLB), i), label="CRLB", color="red")
    plt.tight_layout()
    plt.legend(fontsize=14, loc='upper right')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xlabel("iterations")
    plt.ylabel("Errors")
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plotting results
    sns.set(style="whitegrid")
    colors = ["blue", "yellow", "cyan", "green", "purple", "magenta"]
    for idx, (name, _, _, _) in enumerate(experiment_configs):
        #color = next(plt.gca()._get_lines.prop_cycler)['color']
        ax.plot(np.arange(i), similarity[idx], color=colors[idx], label=name)
    
    #ax.plot(np.arange(i), np.repeat(np.mean(CRLB), i), label="CRLB", color="red")
    plt.tight_layout()
    plt.legend(fontsize=14, loc='lower right')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xlabel("Iterations")
    plt.ylabel("Procrustes Similarity")
    plt.show()
    """
    fig, ax = plt.subplots(figsize=(6, 6)) 

    # Plotting results
    sns.set(style="whitegrid")
    colors = ["blue", "yellow", "cyan", "green", "purple", "magenta"]
    for idx, (name, _, _, _) in enumerate(experiment_configs):
        #color = next(plt.gca()._get_lines.prop_cycler)['color']
        ax.plot(np.arange(i), uncertainty[idx], color=colors[idx], label=name)
    
    #ax.plot(np.arange(i), np.repeat(np.mean(CRLB), i), label="CRLB", color="red")
    plt.tight_layout()
    plt.legend(fontsize=14, loc='lower right')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xlabel("Iterations")
    plt.ylabel("Overall Uncertainties")
    plt.show()



    """ # Print or save the average results
    print("Average RMSE over 10 experiments:", avg_rmse)
    print("Average Uncertainties over 10 experiments:", avg_uncertainties)
    print("Average Similarities over 10 experiments:", avg_similarities) """
    
    """ profiler = cProfile.Profile()
    profiler.enable()
    nbp.iterative_NBP()
    profiler.disable()
    profiler.dump_stats('profile_data.prof')

    import pstats
    p = pstats.Stats('profile_data.prof')
    p.sort_stats('cumulative').print_stats(5) """
    #iterative_NBP(D=network, B=B, X=X_true, anchors=X_true[:a], deployment_area=area, n_particles=p, n_iter=i, k=k, radius=r)
