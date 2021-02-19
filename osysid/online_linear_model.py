import logging

import numpy as np

from .online_model import OnlineModel

logger = logging.getLogger(__name__)


class OnlineLinearModel:
    def __init__(self, n: int, k: int, m: int = None, alpha: float = 1.0):
        """Online Linear Model
        Learn adaptive LTI model

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

    def update(self, x, u, xn, y=None):
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
    def n(self):
        return self._n

    @property
    def k(self):
        return self._k

    @property
    def m(self):
        return self._m

    @property
    def alpha(self):
        return self._alpha

    @property
    def T(self):
        return self._T

    @property
    def ready(self):
        return self._ready

    @property
    def A(self):
        if not self._ready:
            logger.warning(f"Model not ready (have not seen enough data)!")
        return self._f.M[:, : self._n]

    @property
    def B(self):
        if not self._ready:
            logger.warning(f"Model not ready (have not seen enough data)!")
        return self._f.M[:, self._n :]

    @property
    def C(self):
        if not self._m:
            raise Exception(f"No output eqution!")
        if not self._ready:
            logger.warning(f"Model not ready (have not seen enough data)!")
        return self._g.M[:, : self._n]

    @property
    def D(self):
        if not self._m:
            raise Exception(f"No output eqution!")
        if not self._ready:
            logger.warning(f"Model not ready (have not seen enough data)!")
        return self._g.M[:, self._n :]
