from sklearn.decomposition import PCA
import torch

# Author: ORippler (github username)
# Github repo: https://github.com/ORippler/gaussian-ad-mvtec
# With adjustments by JohnnyHopp (github username).

def compute_pca(args, features, variance_threshold = 0.95):
    """Compute pca normalization of teacher features retaining variance.

    Contrary to normal pca, this throws away the features with large
    variance and only keeps the ones with small amounts of variance.
    It is expected that those features will activate on the anomalies.
    """
    mean = [feature.mean(dim=0) for feature in features]

    def fit_level(features):
        pca = PCA(n_components=None).fit(features)
        # Select features above variance_threshold.
        variances = pca.explained_variance_ratio_.cumsum()
        last_component = (variances > variance_threshold).argmax()
        # last_component is the index of the last value needed to reach at
        # least the required explained variance.
        # As the true variance lies somewhere in between [last_component - 1,
        # last_component], we include the whole interval for both pca as
        # well as NPCA based dimensionality reduction
        if args.pca:
            return torch.Tensor(pca.components_[: last_component + 1])
        elif args.npca:
            return torch.Tensor(pca.components_[last_component - 1 :])
        else:
            raise ValueError(
                "either hparams.pca or hparams.npca need to be specified"
            )

    components = [fit_level(level) for level in features]
    
    return mean, components

def pca_reduction(features, pca_mean, pca_components, device):
    """Return pca-reduced features (using the computed PCA)."""
    # Features is training_samples x features x height x width.
    # Unsqueeze batch, height & width.
    demeaned = features - pca_mean.unsqueeze(1).unsqueeze(-1)

    def batched_mul_components(level, components):
        # Cannot use einsum because of unsupported broadcasting.
        # So do a permute to (samples x height x width x features).
        reduced = torch.matmul(  # Batched vector matrix multiply.
            level.permute(0, 2, 3, 1).unsqueeze(-2),
            components.t().unsqueeze(0).unsqueeze(0).unsqueeze(0),
        ).squeeze(
            -2
        )  # Squeeze so this is vector matrix multiply.
        return reduced.permute(0, 3, 1, 2)  # Back to BCHW.

    # This is (X - mean).dot(components.t()).
    return batched_mul_components(demeaned.to(device), pca_components.to(device))

