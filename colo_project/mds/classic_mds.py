import numpy as np
from colo_project.utils.graph_utils import n_hop_distance


class ClassicMDS:
    def __init__(self, dim: int = 2):
        """
        Classic metric MDS: given an NxN distance matrix, embed into R^dim.
        """
        self.dim = dim

    def fit_transform(self, D: np.ndarray) -> np.ndarray:
        """
        D: (N, N) symetric distance matrix
        Returns X_hat: (N, dim) coordinated such that
        ||X_hat[i]-X_hat[j] ~= D[i, j].
        """

        N = D.shape[0]
        D2 = D**2
        J = np.eye(N) - np.ones((N, N)) / N
        B = -0.5 * J @ D2 @ J

        vals, vecs = np.linalg.eigh(B)
        idx = np.argsort(vals)[-self.dim:]
        lambdas = np.maximum(vals[idx], 0.0)
        L = np.diag(np.sqrt(lambdas))
        V = vecs[:, idx]
        self.x_hat = V @ L
        return self.x_hat

    def register_anchors_rigid(
            self, X_hat: np.ndarray,
            anchors_true: np.ndarray) -> np.ndarray:
        # least-squares Procrustes
        from scipy.linalg import orthogonal_procrustes
        num_anchors = len(anchors_true)
        A, s = orthogonal_procrustes(X_hat[:num_anchors], anchors_true)
        X_reg = X_hat @ A
        # translate so anchor centroids match
        shift = anchors_true.mean(axis=0) - X_reg[:num_anchors].mean(axis=0)
        self.x_hat_ab_rigid = X_reg + shift
        return self.x_hat_ab_rigid

    def register_anchors_affine(
            self,
            anchors: np.ndarray,
            anchors_hat: np.ndarray,
            X_hat: np.ndarray
    ):
        anchors = anchors.copy()
        anchors_hat = anchors_hat.copy()
        anchors_hat[1:] -= anchors_hat[0]
        anchors[1:] -= anchors[0]
        Q = np.linalg.pinv(anchors_hat) @ anchors
        T = anchors[0] - anchors_hat[0] @ Q
        self.x_hat_ab_affine = X_hat @ Q + T
        return self.x_hat_ab_affine

    def run_mds(
            self,
            X_true: np.ndarray,
            D: np.ndarray,
            full_D: np.ndarray,
            num_anchors: int,
            use_full_D: bool = False
    ):
        if use_full_D:
            self.mds_dist_graph = full_D
        else:
            hop = 2
            self.mds_dist_graph = n_hop_distance(D, hop)
            expected_nonzero = D.shape[0] * D.shape[1] - D.shape[0]
            while np.count_nonzero(self.mds_dist_graph) < expected_nonzero:
                hop += 1
                self.mds_dist_graph = n_hop_distance(D, hop)

        self.fit_transform(self.mds_dist_graph)
        if num_anchors == 0:
            # Nothing to register against: the embedding is only defined up to
            # rotation/reflection/translation.
            self.x_hat_ab_affine = None
            self.x_hat_ab_rigid = None
            return self.x_hat, None, None

        self.anchors = X_true[:num_anchors]
        self.anchors_hat = self.x_hat[:num_anchors]
        self.register_anchors_affine(
            anchors=self.anchors,
            anchors_hat=self.anchors_hat,
            X_hat=self.x_hat
        )
        self.register_anchors_rigid(
            X_hat=self.x_hat,
            anchors_true=self.anchors
        )

        return self.x_hat, self.x_hat_ab_affine, self.x_hat_ab_rigid


