from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import euclidean_distances
from sklearn.datasets import make_moons
from sklearn.metrics import mean_squared_error
import time
import pprint
import networkx as nx


class ClassicMDS:
    __epsilon = 1e-7
    __noise_mean = 0.0 # noise_mean: mean of gaussian noise. Set to 0.
    __noise_std = 2.0 # noise_std: standard deviation of gaussian noise
    #__seed = 42 # random_seed: seed of random number generator
    def __init__(self,
                 X: np.ndarray = None,
                 D: np.ndarray = None,
                 M: np.ndarray = None,
                 n_samples: int = 100,
                 m_anchors: int = 4,
                 d_dimension: int = 2,
                 radius: float = 30.0,
                 meters: tuple = 50,
                 noise: int = 1,
                 corner_anchors: bool = False,
                 seed: int = None
                 ):
        """
        X: (n_samples, n_features) position matrix. If not given will be generated using
            uniform zero mean distribution
        D: (n_samples, n_samples) Similarity matrix. Euclidean distance matrix or measurement matrix
            calculated from RSSI, ToA, TDoA, AoA etc.
        M: (m_anchor, n_features) anchors matrix with the same dimensions as X.
            If given anchor_count will be at least m_anchors.
        n_samples: number of points in space. Will be used to generate X position matrix.
        m_anchors: number of anchor points will be selected from X
        d_dimension: dimensionality of space
        radius: Used for connectivity information
        meters = area of shape
        noise: Wether noise will be added to distance matrix D or not. Default True means D is noiseless
        and going to be poluted with zero mean gaussian noise. False means no noise will be added to D
        """
        self.seed = seed
        self.m_anchors = m_anchors
        if seed:
            np.random.seed(self.seed)
            
        if D is None:
            self.X_true = X
            if X is None:
                self.X_true = np.random.uniform(-meters, meters, (n_samples, d_dimension))

            assert len(self.X_true.shape) == 2, "X position matrix should be 2D(n, d) shaped"

            if M:
                assert len(M.shape) == len(X.shape), "Should be 2D matrices X and M"
                assert M.shape[1] == self.X_true.shape[1]
                self.X_true = np.vstack([M, self.X_true])
            if corner_anchors:
                anchors = np.fliplr(np.diag(self.X_true.max(axis=0))).T + np.diag(self.X_true.min(axis=0))
                self.X_true = np.vstack([anchors, self.X_true])
                self.m_anchors = anchors.shape[0]

            self.X_mean = self.X_true.mean(axis=0)
            X = self.X = self.X_true #- self.X_mean
            self.n_samples, self.d_dimensions = X.shape

            self.psi = (X @ X.T).diagonal().reshape(-1, 1)
            e = np.ones((self.n_samples, 1))
            D2_temp = self.psi @ e.T - 2 * X @ X.T + e @ self.psi.T # or simply use euclidean_distance(X)
            D2_temp[D2_temp < self.__epsilon] = 0 # for numerical stability
            normal_noise = np.random.normal(self.__noise_mean, self.__noise_std, (self.n_samples, self.n_samples))
            normal_noise -= normal_noise.diagonal()
            self.D = np.sqrt(D2_temp) + normal_noise * noise
        else:
            self.D = D
            self.n_samples = D.shape[0]
            self.d_dimensions = d_dimension

    def solve(self, method="", registration=""):
        pass
    
    def classic_mds(self, D: np.ndarray = None, d_dimension: int = None):
        if D:
            self.D = D
            self.n_samples = self.D.shape[0]
        if d_dimension:
            self.d_dimensions = d_dimension
        self.D2 = self.D**2
        self.I = np.identity(self.n_samples)
        self.e = np.ones((self.n_samples, self.n_samples))
        J = self.I - self.e * (1/self.n_samples)
        JDJ = J @ self.D2 @ J
        B = (-1/2) * JDJ

        eigh = np.linalg.eigh(B)
        eigenvalues, eigenvectors = eigh.eigenvalues, eigh.eigenvectors
        indexes = np.argsort(eigenvalues)[-self.d_dimensions:]
        eigenvalues_diagonal = np.sqrt(np.diag(eigenvalues[indexes]))
        eigenvectors = eigenvectors[:, indexes]
        self.X_hat = X_hat = eigenvectors @ eigenvalues_diagonal
        return self.X_hat

    def least_squares_registration(self, anchors: np.ndarray, anchors_hat: np.ndarray, X_hat: np.ndarray):

        anchors_hat[1:] -= anchors_hat[0]
        anchors[1:] -= anchors[0]
        Q = np.linalg.pinv(anchors_hat) @ anchors
        T = anchors[0] - anchors_hat[0] @ Q
        self.X_hat_ab = X_hat @ Q + T
        return self.X_hat_ab
    
    def plot_results(self, X, X_hat, show_lines=False, show_anchors=False):
        """ W = self.D.copy()
        r = 15
        print(r)
        W[W < r] = 1
        W[W > r] = 0
        pprint.pprint(W)
        g = nx.from_numpy_array(W.copy())
        cliques = nx.find_cliques(g)
        qq = nx.find_cycle(g)
        print(qq)
        
        for c in cliques:
            print(c)
        plt.scatter(X[:, 0], X[:, 1], label="True X")
        for i in range(len(X)):
            for j in range(i, len(X)):
                if i != j and self.D[i, j] < r:
                    plt.plot((X[i, 0], X[j, 0]), (X[i, 1], X[j, 1]))
            plt.annotate(i,X[i]) """
        plt.scatter(X[:, 0], X[:, 1], label="True X")
        plt.scatter(X_hat[:, 0], X_hat[:, 1], label="Predicted Points")
        plt.legend()
        if show_anchors:
            plt.scatter(X[:m, 0], X[:m, 1], "ro")
            plt.scatter(X_hat[:m, 0], X_hat[:m, 1], "go")
        if show_lines:
            for i in range(len(X)):
                plt.plot((X[i, 0], X_hat[i, 0]), (X[i, 1], X_hat[i, 1]), "y--")
        plt.show()

    def RMSE(self, X_true: np.ndarray, X_pred: np.ndarray):
        psi = (X_true @ X_true.T).diagonal().reshape(-1, 1)
        psi_2 = (X_pred @ X_pred.T).diagonal().reshape(-1, 1)
        e = np.ones((self.n_samples, 1))
        error = (psi @ e.T - 2 * X_true @ X_pred.T + e @ psi_2.T).diagonal()
        print(error)
        self.RMSE = np.sqrt(np.mean(error))
        return self.RMSE


n, d, r, m, a = 7, 2, 20, 150, 8 #r kullanılmıyor
noise_mean, noise_std = 0, 5
seed = 93
np.random.seed(seed)

#################

""" X = np.random.uniform(-m, m, (n, d))
noise = np.random.normal(noise_mean, noise_std, (n, n))
noise -= noise.diagonal()
D = euclidean_distances(X)
D_2 = (D + noise)**2
J = np.identity(n) - np.ones((n, n)) * (1/n)
JDJ = J @ D_2 @ J
B = (-1/2) * JDJ
eigh = np.linalg.eigh(B)

eval, evec = eigh.eigenvalues, eigh.eigenvectors
indexes = np.argsort(eval)[-d:]
eval_diag = np.diag(np.sqrt(eval[indexes]))
evec = evec[:, indexes]
X_hat = evec @ eval_diag

anchor = 0
B = X_hat[:a].copy()
B[[range(anchor+1, B.shape[0])]] -= B[anchor]
C = B.copy()

A = X[:a].copy()

A[[range(anchor+1, A.shape[0])]] -= A[anchor]
d = A.copy()

Q = np.linalg.pinv(C) @ d
rotated_Xhat = X_hat @ Q
T = X[anchor] - X_hat[anchor] @ Q
Xhat_ab = rotated_Xhat + T

error = euclidean_distances(X, Xhat_ab)
rmse = np.sqrt(np.mean(error**2))
print(rmse)
print(np.sqrt(mean_squared_error(X, Xhat_ab)))
print(np.sqrt((1/n)*np.sum(error**2)))
print(X[0], Xhat_ab[0])
print((np.sqrt(X[0, 0] - Xhat_ab[0, 0])**2 + (X[0, 1] - Xhat_ab[0, 1])**2))
print(error)

plt.scatter(X[:, 0], X[:, 1])
plt.scatter(Xhat_ab[:, 0], Xhat_ab[:, 1])
plt.show() """
#################

X = np.random.uniform(-m, m, (n, d))

#start = time.time()
#psi = (X @ X.T).diagonal().reshape(-1, 1)

noise = np.random.normal(noise_mean, noise_std, (n, n))
noise -= np.diag(noise.diagonal())
D = euclidean_distances(X) + noise
#mds = ClassicMDS(corner_anchors=False, d_dimension=2, n_samples=n, m_anchors=a, meters=m, noise=0)
mds = ClassicMDS(D=D)
X_hat = mds.classic_mds()
mds.plot_results(X, X_hat)
X_hat_ab = mds.least_squares_registration(anchors=X[:mds.m_anchors].copy(), anchors_hat=X_hat[:mds.m_anchors].copy(), X_hat=X_hat)
mds.plot_results(X, X_hat_ab)
print(mds.RMSE(X, X_hat_ab))

""" D_2 = (D)**2

J = np.identity(n) - np.ones((n, n)) * (1/n)

JDJ = J @ D_2 @ J
JXXJ = -2 * J @ X @ X.T @ J
print(np.allclose(JDJ, JXXJ))


B = (-1/2) * JDJ
B2 = J @ X @ X.T @ J
print(np.allclose(B2, B))

eig = np.linalg.eigh(B)
e_val, e_vec = eig.eigenvalues, eig.eigenvectors
I = e_vec @ e_vec.T
ep = 1e-9
I[I < ep] = 0
print(np.allclose(I, np.identity(n))) # orthogonality check
e_val[e_val < ep] = 0
#print(e_val[e_val > 0].shape[0])


ev = np.diag(e_val)
ev_sqrt = np.sqrt(ev)

B_ = e_vec @ ev @ e_vec.T
B_2 = e_vec @ ev_sqrt @ (e_vec @ ev_sqrt).T
print(B)
print("===================")
print(B_)

#print(B_2[0])
print(np.allclose(B, B2))
print(np.allclose(B, B_))
print(np.allclose(B, B_2))

idxs = np.argsort(e_val)[-d:]
ev_diag = np.sqrt(np.diag(e_val[idxs]))
evec = e_vec[:, idxs]
X_hat = evec @ ev_diag

r = n
plt.scatter(X_hat[:r, 0], X_hat[:r, 1])
plt.scatter(X[:r, 0], X[:r, 1])
#plt.scatter(anchors[:, 0], anchors[:, 1], c="red")
#plt.scatter(X[:a, 0], X[:a, 1], c="red")
for i in range(len(X)):
    plt.plot((X[i, 0], X_hat[i, 0]), (X[i, 1], X_hat[i, 1]), "g--")
plt.show()

end = time.time()
print(f"Time taken calculating relative positions {(end - start)*10**3:.03f}ms")

anchor = 0

B = X_hat[:a].copy()
B[[range(anchor+1, B.shape[0])]] -= B[anchor]
C = B.copy()

A = X[:a].copy()

A[[range(anchor+1, A.shape[0])]] -= A[anchor]
d = A.copy()

Q = np.linalg.pinv(C) @ d
rotated_Xhat = X_hat @ Q
T = X[anchor] - X_hat[anchor] @ Q
Xhat_ab = rotated_Xhat + T
psi_2 = (Xhat_ab @ Xhat_ab.T).diagonal().reshape(-1, 1)
#print(euclidean_distances(X, Xhat_ab)**2)
d_hat_2 = psi@(np.ones((n, 1)).T) - 2 * X @ Xhat_ab.T + np.ones((n, 1))@(psi_2.T)
#print(d_hat_2)
#print(euclidean_distances(X, Xhat_ab)**2 - d_hat_2)
end = time.time()
print(f"Time taken after shifting {(end - start)*10**3:.03f}ms")

#Xhat_ab += 5
#Xhat_ab += X_means
error = euclidean_distances(X, Xhat_ab).diagonal()
#print(error)
rmse = np.sqrt((1/n)*np.sum(error**2))
rmse_2 = np.sqrt(np.mean(d_hat_2.diagonal()))
print(f"RMSE: {rmse} in meters")
print(f"RMSE2: {rmse_2} in meters")
#print(error)
r = n

plt.scatter(Xhat_ab[:r, 0], Xhat_ab[:r, 1])
plt.scatter(X[:r, 0], X[:r, 1])
#plt.scatter(anchors[:, 0], anchors[:, 1], c="red")
#plt.scatter(X[:a, 0], X[:a, 1], c="red")
for i in range(len(X)):
    plt.plot((X[i, 0], Xhat_ab[i, 0]), (X[i, 1], Xhat_ab[i, 1]),"g--") 
plt.show()
 """


""" #import numpy as np
#from matplotlib import pyplot as plt
#from matplotlib.collections import LineCollection

from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances

#n_samples = 20
#seed = np.random.RandomState(42)
#X_true = seed.uniform(10, 190, 2 * n_samples).astype(float)
#X_true = X_true.reshape((n_samples, 2))
# Center the data
#X -= X.mean()
similarities = euclidean_distances(X)

mds = manifold.MDS(
    n_components=2,
    max_iter=1000,
    random_state=seed,
    dissimilarity="precomputed",
    n_jobs=1,
    normalized_stress="auto",
)
pos = mds.fit(similarities).embedding_

# Rescale the data
pos *= np.sqrt((X**2).sum()) / np.sqrt((pos**2).sum())

# Rotate the data
clf = PCA(n_components=2)
X = clf.fit_transform(X)

pos = clf.fit_transform(pos)
#pos = pos @ Q + T

#print(euclidean_distances(X, pos).diagonal())
s = 1
plt.scatter(X[:, 0], X[:, 1], color="navy", label="True Position")
plt.scatter(pos[:, 0], pos[:, 1], color="turquoise", label="MDS")
plt.legend(scatterpoints=1, loc="best", shadow=False)

plt.show()
print(euclidean_distances(Xhat_ab, pos).diagonal()) """


""" # Import numpy for matrix operations
import numpy as np

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
    idx = np.argsort(eig_vals)[::-1] # Sort the eigenvalues in descending order
    eig_vals = eig_vals[idx][:d] # Select the top d eigenvalues
    eig_vecs = eig_vecs[:,idx][:,:d] # Select the corresponding eigenvectors

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
def construct_B(X, W):
    # X is a N x d coordinate matrix[^1^][1]
    # W is a N x N weight matrix
    # The output B is a N x N matrix used in the SMACOF algorithm
    N, d = X.shape
    B = np.zeros((N, N))
    D = euclidean_distances(X) # Calculate the Euclidean distance matrix from X
    for i in range(N):
        for j in range(i+1, N):
            if D[i,j] != 0: # Avoid division by zero
                B[i,j] = B[j,i] = W[i,j] * (D[i,j]**(-2)) * np.dot(X[i] - X[j], X[i] - X[j])
                B[i,i] -= B[i,j]
                B[j,j] -= B[i,j]
    return B

# Define a function to implement the SMACOF algorithm for co-localization
def smacof(D, W, d, max_iter, tol):
    # D is a N x N distance measurement matrix[^1^][1]
    # W is a N x N weight matrix
    # d is the dimension of the output coordinates (usually 2 or 3)
    # max_iter is the maximum number of iterations
    # tol is the tolerance for convergence criterion
    # The output X_hat is a N x d estimated coordinate matrix

    X_hat = classic_mds(D, d)

    # Step 2: Iteratively update X_hat using SMACOF algorithm
    M = construct_M(W) # Calculate the M matrix from W
    stress_old = np.inf # Initialize the old stress value to infinity
    for i in range(max_iter):
        B = construct_B(X_hat, W) # Calculate the B matrix from X_hat and W
        X_hat = np.linalg.pinv(M) @ B @ X_hat # Update X_hat using SMACOF formula
        stress_new = np.sum(W * (D - euclidean_distances(X_hat))**2) # Calculate the new stress value
        if np.abs(stress_new - stress_old) < tol: # Check the convergence criterion
            break # Stop the iteration
        stress_old = stress_new # Update the old stress value
    return X_hat


def bfgs(D, p, X0, max_iter, tol):
    # D is the distance matrix[^1^][1]
    # p is the number of dimensions for the output
    # X0 is the initial guess of the coordinates
    # max_iter is the maximum number of iterations
    # tol is the tolerance for convergence
    
    # Step 1: Define the objective function and its gradient
    def f(X):
        # X is a 3N x 1 vector of coordinates[^3^][3]
        N = len(X) // 3 # number of nodes
        X = X.reshape((N, 3)) # reshape X into a N x 3 matrix[^4^][4]
        sigma = 0 # initialize the objective function value
        for i in range(N):
            for j in range(i+1, N):
                if D[i, j] > 0: # only consider measured distances
                    dij = np.linalg.norm(X[i] - X[j]) # Euclidean distance between node i and j[^5^][5]
                    sigma += (D[i, j] - dij)**2 # add the squared error to the objective function
        return sigma / 2 # return the objective function value
    
    def grad_f(X):
        # X is a 3N x 1 vector of coordinates[^3^][3]
        N = len(X) // 3 # number of nodes
        X = X.reshape((N, 3)) # reshape X into a N x 3 matrix[^4^][4]
        g = np.zeros_like(X) # initialize the gradient matrix
        for i in range(N):
            for j in range(i+1, N):
                if D[i, j] > 0: # only consider measured distances
                    dij = np.linalg.norm(X[i] - X[j]) # Euclidean distance between node i and j[^5^][5]
                    g[i] += (dij - D[i, j]) * (X[i] - X[j]) / dij # update the gradient for node i
                    g[j] += (dij - D[i, j]) * (X[j] - X[i]) / dij # update the gradient for node j
        return g.flatten() # return the gradient vector
    
    # Step 2: Initialize the inverse Hessian matrix and the iteration counter
    H = np.eye(3 * N) # set H to be the identity matrix of size 3N x 3N
    k = 0 # set the iteration counter to zero
    
    # Step 3: Repeat until convergence or maximum iterations reached
    while k < max_iter:
        f_k = f(X0) # evaluate the objective function at current point
        g_k = grad_f(X0) # evaluate the gradient at current point
        
        if np.linalg.norm(g_k) < tol: # check if the gradient is small enough
            break # convergence achieved, exit the loop
        
        h_k = -H.dot(g_k) # compute the search direction using the inverse Hessian matrix
        
        alpha_k = 1 # initialize the step size to be one
        
        c1 = 1e-4 # set the parameter for the Armijo condition
        c2 = 0.9 # set the parameter for the curvature condition
        
        while f(X0 + alpha_k * h_k) > f_k + c1 * alpha_k * g_k.dot(h_k): 
            alpha_k *= c2 # backtracking line search with Armijo condition
        
        s_k = alpha_k * h_k # update s_k
        X1 = X0 + s_k # update X1
        
        f_k1 = f(X1) # evaluate the objective function at new point
        g_k1 = grad_f(X1) # evaluate the gradient at new point
        
        y_k = g_k1 - g_k # update y_k
        
        rho_k = 1 / y_k.dot(s_k) # compute rho_k
        
        H = (np.eye(3 * N) - rho_k * np.outer(s_k, y_k)).dot(H).dot(np.eye(3 * N) - rho_k * np.outer(y_k, s_k)) + rho_k * np.outer(s_k, s_k) 
        # update H using BFGS formula[^7^][7]
        
        X0 = X1 # set X0 to be X1 for next iteration
        
        k += 1 # increment the iteration counter
    
    return X0.reshape((N, p)) # return the final coordinates matrix """

