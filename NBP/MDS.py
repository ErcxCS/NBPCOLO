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


if  __name__ == '__main__':
    n, d, r, m, a = 8, 2, 20, 100, 4 #r kullanılmıyor
    noise_mean, noise_std = 0, 3
    seed = 93
    np.random.seed(seed)
    X = np.random.uniform(-m, m, (n, d))
    X = np.array([
        [m/5, m*1/3],
        [m/5-10, m*2/3-10],
        [m*4/5, m*1/3+15],
        [m*4/5, m*2/3],
        [m/2-10, m/2+10],
        [m*2/5-10, m*2/5],
        [m*2/4-2, m*2/4],
        [m*3/5, m*3/5]
    ])
    X -= X.mean(axis=0)

    #start = time.time()
    #psi = (X @ X.T).diagonal().reshape(-1, 1)

    noise = np.random.normal(noise_mean, noise_std, (n, n))
    noise -= np.diag(noise.diagonal())
    D = euclidean_distances(X) + noise
    print(D)
    #mds = ClassicMDS(corner_anchors=False, d_dimension=2, n_samples=n, m_anchors=a, meters=m, noise=0)
    mds = ClassicMDS(D=D)
    X_hat = mds.classic_mds()
    mds.plot_results(X, X_hat)
    X_hat_ab = mds.least_squares_registration(anchors=X[:mds.m_anchors].copy(), anchors_hat=X_hat[:mds.m_anchors].copy(), X_hat=X_hat)
    mds.plot_results(X, X_hat_ab)
    print(mds.RMSE(X, X_hat_ab))
        #################
