import logging

import numpy as np

logger = logging.getLogger(__name__)


class OnlineModel:
    """OnlineModel is a class that implements online model learning
    The time complexity (multiply–add operation for one iteration) is O(4n^2),
    and space complexity is O(2n^2), where m is the state dimension.

    Algorithm description:
        If the dynamics is z(t) = f(z(t-1), u(t-1)), then we first choose a nonlinear
        observable phi(~, ~), and assume the model can be approximated by
        z(t) = M * phi(z(t-1), u(t-1)).

        Let x(t) = phi(z(t-1), u(t-1)), and y(t) = z(t).
        We would like to learn an adaptive linear model M (a matrix) st y(t) = M * x(t).
        The matrix M is updated recursively by efficient rank-1 updating online algrithm.
        An exponential weighting factor (alpha in (0, 1])) can be used to place more weight on
        recent data.

        At time step t, define two matrix X(t) = [x(1), x(2), ..., x(t)],
        Y(t) = [y(1), y(2),..., y(t)], that contain all the past snapshot pairs,
        where x(t), y(t) are the m dimensional state vector, y(t) = f(x(t)) is
        the image of x(t), f() is the dynamics.

        Here, if there is no control and dynamics is z(t) = f(z(t-1)),
        then x(t), y(t) should be measurements correponding to consecutive
        states z(t-1) and z(t).

    Usage:
        online_model = OnlineModel(n, m, alpha)
        online_model.initialize(X, Y) # optional
        online_model.update(x, y)

    properties:
        M: the model matrix (adaptive and updated online) as in y(t) = M * x(t)
        n: x(t) vector dimension, can be z(t-1) (for z(t) = f(z(t-1))) or phi(z(t-1), u(t-1)) (for z(t) = f(z(t-1), u(t-1)))
            state dimension + control dimension for linear system identification
        m: y(t) vector dimension, z(t) in z(t) = f(z(t-1), u(t-1)) or z(t) = f(z(t-1))
            state dimension for linear system identification
        alpha: weighting factor in (0,1]
        t: number of snapshot pairs processed (i.e., current time step)

    methods:
        initialize(X, Y), initialize online model learning algorithm with first p
                            snapshot pairs stored in (X, Y), this func call is optional
        update(x, y), update online adaptive model when new snapshot pair (x, y)
                            becomes available

    Authors:
        Hao Zhang

    References:
        Zhang, Hao, Clarence W. Rowley, Eric A. Deem, and Louis N. Cattafesta.
        "Online dynamic mode decomposition for time-varying systems."
        SIAM Journal on Applied Dynamical Systems 18, no. 3 (2019): 1586-1609.
    """
    def __init__(self, n: int, m: int, alpha: float = 0.9) -> None:
        """Creat an object for online model learning
        Usage: online_model = OnlineModel(n, m, alpha)
        
        Args:
            n (int): x dimension in model y(t) = M * x(t)
            m (int): y dimension in model y(t) = M * x(t)
            alpha (float, optional): exponential weighting factor in (0, 1], smaller values allows more adaptive learning. Defaults to 0.9.
        """
        # input check
        assert isinstance(n, int) and n >= 1
        assert isinstance(m, int) and m >= 1
        alpha = float(alpha)
        assert alpha > 0 and alpha <= 1

        # initialize parameters
        self._n = n
        self._m = m
        self._alpha = alpha
        self._t = 0
        self._A = np.zeros([m, n])
        self._P = np.zeros([n, n])

        # initialize model
        self._initialize()
        self._ready = False

    def _initialize(self) -> None:
        """Initialize online model with epsilon small (1e-15) ghost snapshot pairs before t=0"""
        epsilon = 1e-15
        self._A = np.random.randn(self._m, self._n)
        self._P = np.identity(self._n) / epsilon

    def initialize(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Initialize online model with first p (p >= max(n, m)) snapshot pairs stored in (X, Y)
        (x(t), y(t)) as in model y(t) = M * x(t)
        Usage: online_model.initialize(X, Y) # this step is optional
        
        Args:
            X (np.ndarray): 2D array, shape (n, p), matrix [x(0), x(1), ..., x(n)]
            Y (np.ndarray): 2D array, shape (m, p), matrix [y(0), y(1), ..., y(n)]
        """
        # input check
        assert X is not None and Y is not None
        X, Y = np.array(X).reshape(self._n, -1), np.array(Y).reshape(self._m, -1)
        assert X.shape[1] == Y.shape[1]

        # necessary condition for over-constrained initialization
        p = X.shape[1]
        assert p >= max(self._n, self._m)
        assert np.linalg.matrix_rank(X) == min(self._n, p)
        
        # initialize with weighted snapshots
        weight = np.sqrt(self._alpha) ** range(p - 1, -1, -1)
        Xhat, Yhat = weight * X, weight * Y
        self._A = Yhat.dot(np.linalg.pinv(Xhat))
        self._P = np.linalg.inv(Xhat.dot(Xhat.T)) / self._alpha
        self._t += p

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        """Update the Model with a new pair of snapshots (x, y)
        y = f(x) is the dynamics/model
        Here, if the (discrete-time) dynamics are given by z(t) = f(z(t-1)),
        then (x,y) should be measurements correponding to consecutive states
        z(t-1) and z(t).
        Usage: online_model.update(x, y)

        Args:
            x (np.ndarray): 1D array, shape (n, ), x(t) in model y(t) = M * x(t)
            y (np.ndarray): 1D array, shape (m, ), y(t) in model y(t) = M * x(t)
        """
        # input check
        assert x is not None and y is not None
        x, y = np.array(x).reshape(-1), np.array(y).reshape(-1)
        assert x.shape[0] == self._n
        assert y.shape[0] == self._m

        # compute P*x matrix vector product beforehand
        Px = self._P.dot(x)
        # compute gamma
        gamma = 1.0 / (1 + x.T.dot(Px))
        # update A
        self._A += np.outer(gamma * (y - self._A.dot(x)), Px)
        # update P, group Px*Px' to ensure positive definite
        self._P = (self._P - gamma * np.outer(Px, Px)) / self._alpha
        # ensure P is SPD by taking its symmetric part
        self._P = (self._P + self._P.T) / 2

        # time step + 1
        self._t += 1

        # mark model as ready
        if self._t >= 2 * max(self._m, self._n):
            self._ready = True

    # ready only properties
    @property
    def M(self) -> np.ndarray:
        """Matrix in model y(t) = M * x(t)

        Returns:
            np.ndarray: [description]
        """
        if not self._ready:
            logger.warning(f"Model not ready (have not seen enough data)!")
        return self._A

    @property
    def m(self) -> int:
        """y dimension in model y(t) = M * x(t)

        Returns:
            int: [description]
        """
        return self._m

    @property
    def n(self) -> int:
        """x dimension in model y(t) = M * x(t)

        Returns:
            int: [description]
        """
        return self._n

    @property
    def alpha(self) -> float:
        """Exponential weighting factor in (0, 1]
        Small value allows more adaptive learning and more forgetting

        Returns:
            float: [description]
        """
        return self._alpha

    @property
    def t(self) -> int:
        """Total number measurements processed
        y(i) = M * x(i)
        (x(i), y(i)), i = 0,1,...,t
        
        Returns:
            int: [description]
        """
        return self._t

    @property
    def ready(self) -> bool:
        """If the model has seen enough data

        Returns:
            bool: [description]
        """
        return self._ready