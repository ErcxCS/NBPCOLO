# RSS and plots of targets and anchors
""" dist = np.linspace(1, 35, 50)
    P_i = -np.linspace(10, 35, 50)
    alpha = 3.15 # path loss exponent
    d0 = 1.15
    epsilon = 1e-9

    RSS = distance_2_RSS(P_i, dist,  alpha, d0, epsilon)
    dist2 = RSS_2_distance(P_i, RSS, alpha, d0, epsilon, sigma=1.12, noise=True)
    RSS2 = distance_2_RSS(P_i, dist2,  alpha, d0, epsilon)
    plt.plot(dist, RSS2)
    plt.xlabel("meters")
    plt.ylabel("RSS")
    plt.show()

    plt.scatter(X_true[a:, 0], X_true[a:, 1], marker="o", c="gray", label="Nt")
    plt.scatter(X_true[:a, 0], X_true[:a, 1], marker="*", c="red", label="Na")
    plt.legend()
    plt.show()

    fig, ax = plt.subplots(1, 2)

    ax[0].scatter(X_true[a:, 0], X_true[a:, 1], marker="o", c="gray", label="Nt")
    ax[1].scatter(X_true[a:, 0], X_true[a:, 1], marker="o", c="gray", label="Nt")
    ax[0].scatter(X_true[:a, 0], X_true[:a, 1], marker="*", c="red", label="Na")
    ax[1].scatter(X_true[:a, 0], X_true[:a, 1], marker="*", c="red", label="Na")
    ax[0].set_title("2-hop")
    ax[1].set_title("1-hop")
    
    nx.draw_networkx_edges(nx.from_numpy_array(graphs["two"]), pos=X_true, ax=ax[0], width=0.5, edge_color="b")
    nx.draw_networkx_edges(nx.from_numpy_array(graphs["one"]), pos=X_true, ax=ax[0], width=0.5)
    nx.draw_networkx_edges(nx.from_numpy_array(graphs["one"]), pos=X_true, ax=ax[1], width=0.5)
    plt.legend()
    plt.show() """


""" SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    label_dict = {
        0: "x0", 1:"x1", 2: "x2", 3: "x3"
    }
    for k, v in label_dict.items():
        label_dict[k] = v.translate(SUB)
    plt.scatter(X_true[:a, 0], X_true[:a, 1], marker="o", c="r", label="anchors")
    plt.scatter(X_true[a:, 0], X_true[a:, 1], marker="o", c="g", label="true")
    nx.draw_networkx_edges(nx.from_numpy_array(network), pos=X_true, width=0.5, arrowstyle='->', arrowsize=10)
    nx.draw_networkx_nodes(nx.from_numpy_array(network), pos=X_true, node_size=300)
    nx.draw_networkx_edge_labels(nx.from_numpy_array(network), pos=X_true, edge_labels={
        (0, 2): r'$m^{n}_{02}(x_2)$',
        (1, 2): r'$m^{n}_{12}(x_2)$',
        (2, 3): r'$m^{n}_{32}(x_2)$',
        })
    nx.draw_networkx_labels(nx.from_numpy_array(network), pos=X_true, labels=label_dict)
    plt.annotate(r'$\phi(x_2)$', xy=X_true[2], xytext=(0, -15), textcoords='offset points')

    plt.title(f"belief of {2}")
    plt.show()
 """

def distance_2_RSS(P_i, D, alpha, d0, epsilon):
    s = 10 * alpha * np.log10((D + epsilon) / d0)
    return P_i - s

def RSS_2_distance(P_i, RSS, alpha, d0, epsilon, sigma: float=.2, noise=True):
    if noise:
        noise = np.random.lognormal(0, sigma=sigma, size=RSS.shape)
        #noise -= np.diag(noise.diagonal())
        symetric_noise = (noise + noise.T) / 2
        RSS += symetric_noise
        
        

    d = d0 * 10 ** ((P_i - RSS) / (10 * alpha)) + epsilon
    #d -= d.diagonal()
    return d


# plotting of neigbors vs error and procrustes distance
""" neighbour_count = [np.count_nonzero((u > 0) & (u < radius)) for u in D]

            anchor_count = [np.count_nonzero((u[:n_anchors] > 0) & (u[:n_anchors] < radius)) for u in D]
            errors = ERR(weighted_means, X[n_anchors:])
            # Create a scatter plot of neighbour count vs error
            plt.scatter(neighbour_count[n_anchors:], errors)

            # Fit a polynomial curve of degree 2 to the data
            z = np.polyfit(neighbour_count[n_anchors:], errors, 2)
            p = np.poly1d(z)

            # Plot the curve on the same figure
            x = np.linspace(min(neighbour_count[n_anchors:]), max(neighbour_count[n_anchors:]), 100)
            plt.plot(x, p(x), color='red')
            print(f"anchor counts: {anchor_count}")
            # Add labels and title
            plt.xlabel('Neighbour count')
            plt.ylabel('Error')
            plt.title('Neighbour count vs error with polynomial fit')
            plt.show()

            Z, T, d = procrustes(X, estimates)

            print(f"Procrustes distance: {d:.2f}") """



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