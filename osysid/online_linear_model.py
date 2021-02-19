import logging

import numpy as np

from .online_model import OnlineModel

logger = logging.getLogger(__name__)


class OnlineLinearModel:
    def __init__(self, n: int, k: int, m: int=None, alpha: float=1.0):
        """Online Linear Model

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
        self.n = n
        self.k = k
        self.m = m
        self.alpha = alpha
        
        # additional parameters
        self.T = 0
        self.ready = False
        
        # initialize model
        self._f = OnlineModel(n, n + k) # encodes A and B
        if self.m:
            logger.info("Learn x(t+1) = A * x(t) + B * u(t), y(t) = C * x(t) + D * u(t)")
            self._g = OnlineModel(m, n + k) # encodes C and D
        else:
            logger.info("No output eqution, only learn x(t+1) = A * x(t) + B * u(t)")
            
    def update(self, x, u, xn, y=None):
        # input check
        assert x.shape[0] == self.n
        assert u.shape[0] == self.k
        assert xn.shape[0] == self.n
        if y is None:
            assert self.m is None
        if self.m:
            assert y.shape[0] == self.m
            
        # update f
        z = np.concatenate((x, u))
        self._f.update(z, xn)
        
        # update g if needed
        if self.m:
            self._g.update(z, y)
        
        # timestep
        self.T += 1
        
        # mark model as ready
        if self.T >= 2 * max(self.n, self.n + self.k, self.m):
            self.ready = True

    # can only get A, but can not set A
    @property
    def A(self):
        if not self.ready:
            logger.warning(f"Model not ready (have not seen enough data)!")
        return self._f.A[:, :self.n]

    @property
    def B(self):
        if not self.ready:
            logger.warning(f"Model not ready (have not seen enough data)!")
        return self._f.A[:, self.n:]
    
    @property
    def C(self):
        if not self.m:
            raise Exception(f"No output eqution!")
        if not self.ready:
            logger.warning(f"Model not ready (have not seen enough data)!")
        return self._g.A[:, :self.n]
    
    @property
    def D(self):
        if not self.m:
            raise Exception(f"No output eqution!")
        if not self.ready:
            logger.warning(f"Model not ready (have not seen enough data)!")
        return self._g.A[:, self.n:]