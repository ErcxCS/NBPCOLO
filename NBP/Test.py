# Import numpy for matrix operations
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from sklearn import manifold

# Define a function to calculate the Euclidean distance matrix from a coordinate matrix
def euclidean_distances(X):
    # X is a N x d matrix, where N is the number of nodes and d is the dimension
    # The output D is a N x N matrix, where D[i,j] is the Euclidean distance between X[i] and X[j]
    N, d = X.shape
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            D[i,j] = D[j,i] = np.linalg.norm(X[i] - X[j])
    return D

# Define a function to calculate the B matrix from a distance matrix
def double_centering(D):
    # D is a N x N distance matrix[^1^][1]
    # The output B is a N x N matrix that represents the dissimilarities between nodes
    N = D.shape[0]
    e = np.ones(N)
    I = np.eye(N)
    J = I - (1/N) * np.outer(e, e) # J is the centering matrix
    B = -0.5 * J @ (D**2) @ J # B is the double centered matrix
    return B

# Define a function to implement the Classic MDS algorithm for co-localization
def classic_mds(D, d):
    # D is a N x N distance measurement matrix[^1^][1]
    # d is the dimension of the output coordinates (usually 2 or 3)
    # The output X_hat is a N x d estimated coordinate matrix

    # Step 1: Calculate the B matrix from D using double centering transformation
    B = double_centering(D)
    # Step 2: Perform eigenvalue decomposition on B and select the top d eigenvalues and eigenvectors
    eig_vals, eig_vecs = np.linalg.eigh(B) # Perform eigenvalue decomposition on B
    eigh = np.linalg.eigh(B)
    idx = np.argsort(eig_vals)[-d:] # Sort the eigenvalues in descending order
    eig_vals = eig_vals[idx] # Select the top d eigenvalues
    eig_vecs = eig_vecs[:,idx] # Select the corresponding eigenvectors

    # Step 3: Calculate X_hat from eigenvectors and eigenvalues using square root transformation
    X_hat = eig_vecs @ np.diag(np.sqrt(eig_vals))

    return X_hat


# Define a function to calculate the M matrix from a weight matrix
def construct_M(W):
    # W is a N x N weight matrix[^1^][1]
    # The output M is a N x N matrix used in the SMACOF algorithm
    N = W.shape[0]
    M = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            M[i,j] = M[j,i] = -W[i,j]
            M[i,i] += W[i,j]
            M[j,j] += W[i,j]
    return M

# Define a function to calculate the B matrix from a coordinate matrix and a weight matrix
def construct_B(D, dis, W):
        dis[dis == 0] = 1e-7
        ratio = D / dis
        B = -ratio
        B[np.arange(len(B)),np.arange(len(B))] += ratio.sum(axis=1)
        return B * W

# Define a function to implement the SMACOF algorithm for co-localization
def smacof(D, W, k, d, max_iter, tol):
    # D is a N x N distance measurement matrix
    # W is a N x N weight matrix
    # d is the dimension of the output coordinates (usually 2 or 3)
    # max_iter is the maximum number of iterations
    # tol is the tolerance for convergence criterion
    # The output X_hat is a N x d estimated coordinate matrix

    if k == 1:
        X_hat = classic_mds(D, d)
    else:
        X_hat = np.random.normal(size=(D.shape[0], d))
    # Step 2: Iteratively update X_hat using SMACOF algorithm
    M = construct_M(W) # Calculate the M matrix from W
    stress_old = np.inf # Initialize the old stress value to infinity

    for i in range(max_iter):
        dis = euclidean_distances(X_hat)
        stress_new = (W * (dis - D) ** 2).sum()
        B = construct_B(D, dis, W) # Calculate the B matrix from X_hat and W
        X_hat = np.linalg.pinv(M) @ B @ X_hat # Extra matmul because of W
        #dis = np.sqrt((X_hat**2).sum(axis=1)).sum()
        #print(f"Stress: {stress_new}")
        if np.abs((stress_new - stress_old)) < tol: # Check the convergence criterion
            break # Stop the iteration


        stress_old = stress_new # Update the old stress value

    return X_hat, stress_old


def spectral_Test(distance_matrix : np.ndarray, d_dim = 2, r = 800):
    A = distance_matrix.copy()
    A[A > r] = 0
    #A = (distance_matrix < r).astype(int)
    A = distance_matrix.max() - distance_matrix
    #A -= np.diag(A.diagonal())

    G = nx.from_numpy_array(A)
    nx.draw_spectral(G)
    e_vecs = spectral(A, 2)
    #nx_result = np.array(list(nx.spectral_layout(G).values()))
    return e_vecs

def spectral(A: np.ndarray, d: int = 2):
    n_nodes = A.shape[0]
    D = np.identity(n_nodes, dtype=A.dtype) * np.sum(A, axis=1)
    L = D - A
    e_vals, e_vecs = np.linalg.eig(L)
    idxs = np.argsort(e_vals)[1 : d + 1]
    e_vals = e_vals[idxs]
    e_vecs = np.real(e_vecs[:, idxs])

    # scale
    e_vecs -= e_vecs.mean(axis=0)
    lim = np.abs(e_vecs).max()
    scale = 1
    if lim > 0:
        e_vecs *= scale / lim
    return e_vecs 



def ls_registration(A, B):
    # A is the absolute coordinate matrix of anchor nodes
    # B is the relative coordinate matrix of anchor nodes
    # Both A and B are m x d matrices, where m is the number of anchor nodes
    
    # Step 1: Compute the matrix C and d
    
    #C = B.copy()
    #C[1:] -= C[0]
    #d = A.copy()
    #d[1:] -= d[0]
    C = B[1:] - B[0] # subtract the first row of B from the rest
    d = A[1:] - A[0] # subtract the first row of A from the rest


    # Step 2: Solve for the rotation matrix Q by least squares
    Q = np.linalg.lstsq(C, d, rcond=None)[0] # use numpy's lstsq function
    # Step 3: Solve for the translation vector T
    T = A[0] - B[0].dot(Q) # use the first rows of A and B

    return Q, T # return Q and T

def plot_results(X, X_hat, a, ax, show_lines=False, show_anchors=False, alg=""):

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
    return np.random.uniform(-m, m, (n, d))

def rmse(X, X_hat_ab):
    # X is the true coordinates matrix
    # X_hat_ab is the estimated coordinates matrix
    
    # Compute the Euclidean distance for each node
    error = np.sqrt(np.sum((X - X_hat_ab)**2, axis=1))
    # Compute the RMSE
    return np.sqrt(np.mean(error**2))

def log(**kwargs):
    for k, v in kwargs.items():
        print(f"{k}: {v} rmse")
    print("######################")

def cliques(X: np.ndarray, D: np.ndarray, radius):
    W = D.copy()
    W[W > radius] = 0
    plt.scatter(X[:, 0], X[:, 1], label="True X")
    print(W)
    for clique in nx.clique.find_cliques(nx.from_numpy_array(W)):
        #print(clique)
        if (len(clique) > 2):
            print(W[clique])
        for i in range(len(clique)):
            plt.annotate(clique[i], X[clique[i]])
            for j in range(i, len(clique)):
                plt.plot((X[clique[i], 0], X[clique[j], 0]), (X[clique[i], 1], X[clique[j], 1]))
    plt.show()
    #print(nx.find_cycle(nx.from_numpy_array(W)))
    return 0


def test_mds(n_samples: int = 50,
             d_dimensions: int = 2,
             radius: float = 0.0,
             meters: int = 15,
             anchors: int = 4,
             k_test: int = 5,
             seed: int = None,
             k: int = 1
            ):
    
    np.random.seed(seed)
    for i in range(k_test):
        X = generate_X_points(meters, n_samples, d_dimensions)
        noise = np.random.normal(0, 2, (n_samples, n_samples))
        noise -= np.diag(noise.diagonal())
        symetric_nosie = (noise + noise.T) / 2
        D = euclidean_distances(X) + symetric_nosie*0
        results = dict()
        #print(D)

        ### Classic MDS
        X_hat = classic_mds(D, d_dimensions)
        # Registration
        Q, T = ls_registration(X.copy()[:anchors], X_hat.copy()[:anchors])
        X_hat_ab = X_hat @ Q + T
        # RMSE
        results["classic_rmse"] = rmse(X, X_hat_ab)

        ### my SMACOF
        #W = (D < radius).astype(int)
        #W -= np.diag(W.diagonal())
        W = np.ones((D.shape))
        #print(W)

        best_pos, best_stress = None, None
        for i in range(k):
            pos, stress = smacof(D, W, k, d_dimensions, 1000, 1e-3)
            if best_stress is None or stress < best_stress:
                best_pos = pos
                best_stress = stress
        X_hat = best_pos
        # Registration
        Q, T = ls_registration(X.copy()[:anchors], X_hat.copy()[:anchors])
        smacof_X_hat_ab = X_hat @ Q + T
        # RMSE
        results["smacof_rmse"] = rmse(X, smacof_X_hat_ab)

        ### scikit-learn SMACOF/MDS
        mds = manifold.MDS(
            n_components=2,
            max_iter=1000,
            random_state=seed,
            dissimilarity="precomputed",
            n_jobs=1,
            metric=True,
            normalized_stress="auto"
        )
        X_hat = mds.fit_transform(D)
        # Registration
        Q, T = ls_registration(X.copy()[:anchors], X_hat.copy()[:anchors])
        sk_smacof_X_hat_ab = X_hat @ Q + T
        # RMSE
        results["sk_smacof_rmse"] = rmse(X, sk_smacof_X_hat_ab)

        # Spectral Layout
        e_vecs = spectral_Test(D, 2, radius)
        Q, T = ls_registration(X.copy()[:anchors], e_vecs.copy()[:anchors])
        e_vecs_ab = e_vecs @ Q + T
        results["spectral"] = rmse(X, e_vecs_ab)

        fig, (ax1, ax2) = plt.subplots(1, 2)
        plot_results(X, X_hat_ab, a=4, ax=ax1)
        plot_results(X, e_vecs_ab, a=4, ax=ax2, show_lines=True)
        plt.show()

        # Logging
        clique_hat = cliques(X, D, radius)
        log(**results)

seed = 42
np.random.seed(seed)
n, d, r, m, a, k = 50, 2, 25, 75, 3, 1
test_mds(n_samples=n, d_dimensions=d, radius=r, meters=m, anchors=a, k_test=1, seed=seed, k=1)


#Plotting
""" fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
plot_results(X=X, X_hat=X_hat_ab, a=a, ax=ax1, show_lines=False, show_anchors=False, alg="Classic")
plot_results(X=X, X_hat=smacof_X_hat_ab, a=a, ax=ax2, show_lines=False, show_anchors=False, alg="SMACOF")
plot_results(X=X, X_hat=SK_smacof_X_hat_ab, a=a, ax=ax3, show_lines=False, show_anchors=False, alg="SK_SMACOF") """


#plot_results(X, X_hat, a, ax1, alg="SMACOF")
#Q, T = ls_registration(X.copy()[:a], X_hat.copy()[:a])
#X_hat_ab = X_hat @ Q + T
#print(f"MDS rmse: {rmse(X, X_hat_ab)}")
#plot_results(X, X_hat_ab, a, ax2, alg="MDS")
#plt.show()

""" adj = (D < r).astype(int)
adj -= np.diag(adj.diagonal())
print(adj)
G = nx.from_numpy_array(adj)

fig, (ax1, ax2) = plt.subplots(1, 2)
X_hat = np.array(list(nx.spectral_layout(G).values())) * 100
plot_results(X, X_hat, a=4, ax=ax1, alg="non-rotate")

Q, T = ls_registration(X[:a].copy(), X_hat[:a].copy())
print(euclidean_distances(X_hat))
X_hat_ab = X_hat @ Q + T
print("====================")
print(euclidean_distances(X_hat_ab))
plot_results(X, X_hat_ab, a=4, ax=ax2, alg="rotated")

print(rmse(X, X_hat_ab))
print(len(G))
pos = nx.spectral_layout(G) """
""" for k, v in pos.items():
    pos[k] = pos[k] * 100 """

#nx.draw(G, pos, with_labels=True, ax=ax2)
#plt.show()
#print(np.array(list(nx.spectral_layout(G).values())))




""" from sklearn import manifold
mds = manifold.MDS(
    n_components=3,
    max_iter=1000,
    random_state=seed,
    dissimilarity="precomputed",
    n_jobs=1,
    normalized_stress="auto",
)
pos = mds.fit(D).embedding_
Q, T = ls_registration(X.copy()[:a], pos.copy()[:a])
pos_res = pos @ Q + T
print(f"SMACOF rmse: {rmse(X, pos_res)}")
plot_results(X, pos_res, a, alg="Pos")
print(pos_res)
print(X)
 """
