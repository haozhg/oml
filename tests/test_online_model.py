import numpy as np
from osysid import OnlineModel

np.random.seed(20210218)


def update(A, eps=1e-6):
    return A + eps * np.random.randn(*A.shape)


def test_online_model():
    # m is y dimension
    for m in range(2, 10):
        n = 2 * m # x dimension
        T = 16 * m # total number of measurements
        print(f"m={m}, n={n}, T={T}")

        # true model, slowly varying in time
        A = np.random.randn(m, n)

        # online model learning
        # no need to initialize
        online_model = OnlineModel(n, m, alpha=0.5)
        for t in range(T):
            x = np.random.randn(n)
            y = A.dot(x)
            online_model.update(x, y)
            if online_model.ready:
                assert np.linalg.norm(online_model.M - A) / (m * n) < 1e-3

            # update time-varying model
            A = update(A)


if __name__ == "__main__":
    test_online_model()
