import logging

import numpy as np

logger = logging.getLogger(__name__)


class OnlineModel:
    """OnlineModel is a class that implements online model learning
    The time complexity (multiply–add operation for one iteration) is O(4n^2),
    and space complexity is O(2n^2), where n is the state dimension.

    Algorithm description:
        If the dynamics is z(k) = f(z(k-1),u(k-1)), then we first choose a nonlinear
        observable phi(~, ~), and assume the model can be approximated by
        z(k) = M * phi(z(k-1), u(k-1)).

        Let x(k) = phi(z(k-1),u(k-1)), and y(k) = z(k).
        We would like to learn an adaptive linear model M (a matrix) st y(k) = M * x(k).
        The matrix M is updated recursively by efficient rank-1 updating online algrithm.
        An exponential alpha factor can be used to place more weight on
        recent data.

        At time step k, define two matrix X(k) = [x(1),x(2),...,x(k)],
        Y(k) = [y(1),y(2),...,y(k)], that contain all the past snapshot pairs,
        where x(k), y(k) are the n dimensional state vector, y(k) = f(x(k)) is
        the image of x(k), f() is the dynamics.

        Here, if there is no control and dynamics is z(k) = f(z(k-1)),
        then x(k), y(k) should be measurements correponding to consecutive
        states z(k-1) and z(k).

    Usage:
        online_model = OnlineModel(n, alpha)
        online_model.initialize(X, Y) # optional
        online_model.update(x, y)

    properties:
        M: the model matrix (adaptive and updated online)
        n: state dimension
        q: observable vector dimension, state dimension + control dimension for
           linear system identification case
        alpha: weight factor in (0,1]
        T: number of snapshot pairs processed (i.e., current time step)

    methods:
        initialize(X, Y), initialize online model learning algorithm with first m
                            snapshot pairs stored in (X, Y), this func call is optional
        update(x,y), update online adaptive model when new snapshot pair (x,y)
                            becomes available

    Authors:
        Hao Zhang

    References:
        Zhang, Hao, Clarence W. Rowley, Eric A. Deem, and Louis N. Cattafesta.
        "Online dynamic mode decomposition for time-varying systems."
        SIAM Journal on Applied Dynamical Systems 18, no. 3 (2019): 1586-1609.
    """

    def __init__(self, n: int, q: int, alpha: float = 1.0):
        """
        Creat an object for online model learning
        Usage: online_model = OnlineModel(n,alpha)
        """
        # input check
        assert isinstance(n, int) and n >= 1
        assert isinstance(q, int) and q >= 1
        assert alpha > 0 and alpha <= 1

        # initialize parameters
        self._n = n
        self._q = q
        self._alpha = alpha
        self._T = 0
        self._A = np.zeros([n, q])
        self._P = np.zeros([q, q])

        # initialize model
        self._initialize()
        self._ready = False

    def _initialize(self):
        """Initialize online model with epsilon small (1e-15) ghost snapshot pairs before t=0"""
        epsilon = 1e-15
        self._A = np.random.randn(self._n, self._q)
        self._P = np.identity(self._q) / epsilon

    def initialize(self, X, Y):
        """Initialize online model with first m (m >= n, q) snapshot pairs stored in (X, Y)
        Usage: online_model.initialize(X, Y)
        """
        # input check
        assert X is not None and Y is not None
        X, Y = np.array(X).reshape(self._n, -1), np.array(Y).reshape(self._n, -1)
        assert X.shape[0] == self._n
        assert Y.shape[0] == self._q
        assert X.shape[1] == Y.shape[1]

        # necessary condition for over-constrained initialization
        m = X.shape[1]
        assert m >= max(self._n, self._q)
        assert np.linalg.matrix_rank(X) == min(self._n, m)

        # initialize with weighted snapshots
        weight = np.sqrt(self._alpha) ** range(m - 1, -1, -1)
        Xhat, Yhat = weight * X, weight * Y
        self._A = Yhat.dot(np.linalg.pinv(Xhat))
        self._P = np.linalg.inv(Xhat.dot(Xhat.T)) / self._alpha
        self._T += m

    def update(self, x, y):
        """Update the Model with a new pair of snapshots (x,y)
        y = f(x) is the dynamics/model
        Here, if the (discrete-time) dynamics are given by z(k) = f(z(k-1)),
        then (x,y) should be measurements correponding to consecutive states
        z(k-1) and z(k).
        Usage: online_model.update(x, y)
        """
        # input check
        assert x is not None and y is not None
        x, y = np.array(x).reshape(-1), np.array(y).reshape(-1)
        assert x.shape[0] == self._q
        assert y.shape[0] == self._n

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
        self._T += 1

        # mark model as ready
        if self._T >= 2 * max(self._n, self._q):
            self._ready = True

    # can only get A, but can not set A
    @property
    def M(self):
        if not self._ready:
            logger.warning(f"Model not ready (have not seen enough data)!")
        return self._A

    @property
    def n(self):
        return self._n

    @property
    def q(self):
        return self._q

    @property
    def alpha(self):
        return self._alpha

    @property
    def T(self):
        return self._T
