import logging

import numpy as np

from ._online_model import OnlineModel

logger = logging.getLogger(__name__)


class OnlineLinearModel:
    """
    # Unknown dynamical system
    Suppose we have a (discrete) nonlinear and/or time-varying [dynamical system](https://en.wikipedia.org/wiki/State-space_representation), 
    and the state space representation is
    - x(t+1) = f(t, x(t), u(t))
    - y(t) = g(t, x(t), u(t))

    where t is (discrete) time, x(t) is state vector, u(t) is control (input) vector, y(t) is observation (output) vector. 
    f(~, ~, ~) and g(~, ~, ~) are unknown vector-valued nonlinear functions.

    - It is assumed that we have measurements x(t), u(t), y(t) for t = 0,1,...T. 
    - However, we do not know functions f and g. 
    - We aim to learn a model for the unknown dynamical system from measurement data up to time T.
    - We want to the model to be updated efficiently in real-time as new measurement data becomes available.

    # Online linear model learning
    We would like to learn an adaptive [linear model](https://en.wikipedia.org/wiki/State-space_representation)
    - x(t+1) = A x(t) + B u(t)
    - y(t) = C x(t) + D u(t)

    that fits/explains the observation optimally. By Taylor expansion approximation, any nonlinear and/or 
    time-varying system is linear locally. There are many powerful tools for linear control, 
    e.g, [Linear Quadratic Regulator](https://en.wikipedia.org/wiki/Linear%E2%80%93quadratic_regulator), 
    [Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter). 
    However, to accurately approximate the original (unknown) dynamical system, we need to update this linear 
    model efficiently in real-time whenever new measurement becomes available.

    This problem can be formulated as an optimization problem, and at each time step t we need to solve a 
    related but slightly different optimization problem. The optimal algorithm is achived through efficient 
    reformulation of the problem. 

    """
    def __init__(self, n: int, k: int, m: int = None, alpha: float = 1.0) -> None:
        """Online Linear Model
        Efficiently learn adaptive linear model from data in real-time
        
        Args:
            n (int): state dimension
            k (int): control (input) dimension
            m (int): observation (output) dimension
            alpha (float, optional): exponentiall weighting factor in system identification. Defaults to 1.0.
        """
        # input check
        assert isinstance(n, int) and n >= 1
        assert isinstance(k, int) and k >= 1
        assert m is None or (isinstance(m, int) and m >= 1)
        assert alpha > 0 and alpha <= 1

        # set parameters
        self._n = n
        self._k = k
        self._m = m
        self._alpha = alpha

        # additional parameters
        self._T = 0
        self._max = max(self._n, self._n + self._k)
        if self._m:
            self._max = max(self._max, self._m)
        self._max = 2 * self._max
        self._ready = False

        # initialize model
        self._f = OnlineModel(n, n + k)  # encodes A and B
        if self._m:
            logger.info("Learn x(t+1) = A * x(t) + B * u(t), y(t) = C * x(t) + D * u(t)")
            self._g = OnlineModel(m, n + k)  # encodes C and D
        else:
            logger.info("No output eqution, only learn x(t+1) = A * x(t) + B * u(t)")
            
    def update(self, x: np.ndarray, u: np.ndarray, xn: np.ndarray, y: np.ndarray =None) -> None:
        """Update model wrt new measurement (state x, control u, next state xn, optinal observation y)

        Args:
            x (np.ndarray): state x(t), 1D array, shape (n, )
            u (np.ndarray): control u(t), 1D array, shape (k, )
            xn (np.ndarray): new state x(t+1), 1D array, shape (n, )
            y (np.ndarray, optional): observation y(t), 1D array, shape (m, ). Defaults to None.
        """
        # input check
        assert x.shape[0] == self._n
        assert u.shape[0] == self._k
        assert xn.shape[0] == self._n
        if y is None:
            assert self._m is None
        if self._m:
            assert y.shape[0] == self._m

        # update f
        z = np.concatenate((x, u))
        self._f.update(z, xn)

        # update g if needed
        if self._m:
            self._g.update(z, y)

        # timestep
        self._T += 1

        # mark model as ready
        if self._T >= self._max:
            self._ready = True

    # no setter
    @property
    def A(self) -> np.ndarray:
        """A in state space representation

        Returns:
            np.ndarray: [description]
        """
        if not self._ready:
            logger.warning(f"Model not ready (have not seen enough data)!")
        return self._f.M[:, : self._n]

    @property
    def B(self) -> np.ndarray:
        """B in state space representation

        Returns:
            np.ndarray: [description]
        """
        if not self._ready:
            logger.warning(f"Model not ready (have not seen enough data)!")
        return self._f.M[:, self._n :]

    @property
    def C(self) -> np.ndarray:
        """C in state space representation

        Raises:
            Exception: if no output eqution

        Returns:
            np.ndarray: [description]
        """
        if not self._m:
            raise Exception(f"No output eqution!")
        if not self._ready:
            logger.warning(f"Model not ready (have not seen enough data)!")
        return self._g.M[:, : self._n]

    @property
    def D(self) -> np.ndarray:
        """D in state space representation

        Raises:
            Exception: if no output eqution

        Returns:
            np.ndarray: [description]
        """
        if not self._m:
            raise Exception(f"No output eqution!")
        if not self._ready:
            logger.warning(f"Model not ready (have not seen enough data)!")
        return self._g.M[:, self._n :]

    @property
    def n(self) -> int:
        """State x(t) dimension as in state space representation

        Returns:
            int: [description]
        """
        return self._n

    @property
    def k(self) -> int:
        """Control (input) u(t) dimension as in state space representation

        Returns:
            int: [description]
        """
        return self._k

    @property
    def m(self) -> int:
        """Observation (output) y(t) dimension as in state space representation

        Returns:
            int: [description]
        """
        return self._m

    @property
    def alpha(self) -> float:
        """Exponential weighting factor in (0, 1]
        Small value allows more adaptive learning and more forgetting

        Returns:
            float: [description]
        """
        return self._alpha

    @property
    def T(self) -> int:
        """Total number measurements processed
        
        Returns:
            int: [description]
        """
        return self._T

    @property
    def ready(self) -> bool:
        """If the model has seen enough data

        Returns:
            bool: [description]
        """
        return self._ready