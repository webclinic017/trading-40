import numpy as np

np.random.seed(0)


class BrownianMotion:
    """
    A Brownian Motion is a continuous process such that its increments for any time scale are drawn from a normal distribution.

    Take a continuous function of t: W = W(t).
    W is a Brownian motion (Wiener Process) if:
    1. W(0) = 0
    2. For all t_0 < t_1 < ... < t_n,  the increments:

        W(t_1) - W(t_0), ... W(t_n) - W(t_[n-1]) are independent and normally distributed.
        --> Normal(mean = 0, var = t_[i+1] - t_i) ,  i.e. var is the time difference.

    If the increments are defined on a unit of time: var = 1.

    Example usage:
        bm = BrownianMotion()
        bm.W(4)
    """

    def __init__(self, mean: float=0.0, std_dev: float=1.0):
        self.mean = mean
        self.std_dev = std_dev

    def dW(self, num_samples: int):
        return np.random.normal(loc=self.mean, scale=self.std_dev, size=num_samples)

    def W(self, num_samples: int):
        dW = self.dW(num_samples)

        # Integrate by cumulative sum
        dW_sum = dW.cumsum()

        # Prepend W(0) = 0 and trim to `num_samples`.
        return np.insert(dW_sum, 0, 0)[:-1]
