import pandas as pd
from algo.models.sde.ornstein_uhlenbeck_model import OrnsteinUhlenbeck
from algo.models.sde.ornstein_uhlenbeck_model_optimisation import OUParams


if __name__ == "__main__":
    ou_params = OUParams(
        theta=0.1405849503082211,
        mu=80.49491464625557,
        sigma=0.12652352136635514,
        sigma_sq=0.016008201458942526,
        sums=(154.57621525792626, 154.59653199802102, 21.629070312663373, 21.634927968936843, 21.630477363692105)
    )

    test_path = "/algo/models/sde/data/ou_test.csv"
    x = pd.read_csv(test_path).to_numpy()[:, 0]

    ou_model = OrnsteinUhlenbeck(X_0=x[0], theta=ou_params.theta, k=ou_params.mu, sigma=ou_params.sigma)
    ou_process_simulated = ou_model(num_samples=len(x))
