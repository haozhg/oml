import numpy as np
from osysid import OnlineModel

np.random.seed(20210218)


def update(A, eps=1e-9):
    return A + eps * np.random.randn(*A.shape)


def test_online_model():
    # n is y dimension
    for n in range(2, 10):
        q = 2 * n # x dimension
        T = 16 * n # total number of measurements
        print(f"{n=}, {q=}, {T=}")

        # true model, slowly varying in time
        A = np.random.randn(n, q)

        # online model learning
        # no need to initialize
        online_model = OnlineModel(n, q, alpha=0.5)
        w = 1e-6
        for t in range(T):
            x = np.random.randn(q)
            y = A.dot(x)
            online_model.update(x, y)
            if t >= 2 * q:
                assert np.linalg.norm(online_model.M - A) / (n * q) < 1e-3

            # update time-varying model
            A = update(A)


if __name__ == "__main__":
    test_online_model()
