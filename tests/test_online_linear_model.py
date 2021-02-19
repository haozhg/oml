import logging

import numpy as np
from osysid import OnlineLinearModel

np.random.seed(20210218)

logger = logging.getLogger(__name__)


def update(A, eps=1e-9):
    return A + eps * np.random.randn(*A.shape)


def test_online_linear_model():
    for n in range(2, 10):
        k = n // 2
        m = n // 2
        print(f"{n=},{k=},{m=}")
        T = 16 * n

        # true model, slowly varying in time
        A = np.random.randn(n, n)
        B = np.random.randn(n, k)
        C = np.random.randn(m, n)
        D = np.random.randn(m, k)

        # online linear model learning
        # no need to initialize
        olm = OnlineLinearModel(n, k, m, alpha=0.5)
        for i in range(T):
            print(f"{i=}")
            # initial condition
            x = np.random.randn(n)
            u = np.random.randn(k)

            # state update
            xn = A.dot(x) + B.dot(u)
            y = C.dot(x) + D.dot(u)

            # update model est
            olm.update(x, u, xn, y)
            if i >= 2 * max(n, n + k, m):
                assert np.linalg.norm(olm.A - A) / (n * n) < 1e-3
                assert np.linalg.norm(olm.B - B) / (n * k) < 1e-3
                assert np.linalg.norm(olm.C - C) / (m * n) < 1e-3
                assert np.linalg.norm(olm.D - D) / (m * k) < 1e-3

            # update time-varying model
            A = update(A)
            B = update(B)
            C = update(C)
            D = update(D)


if __name__ == "__main__":
    test_online_linear_model()
