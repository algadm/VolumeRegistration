import torch
import torch.nn.functional as F

def soft_histogram(x, num_bins, min_val, max_val, sigma=1.0):
    """
    Compute soft histograms for differentiability.
    :param x: Input tensor (flattened image).
    :param num_bins: Number of histogram bins.
    :param min_val: Minimum value for histogram range.
    :param max_val: Maximum value for histogram range.
    :param sigma: Gaussian smoothing for soft bins.
    :return: Soft histogram tensor.
    """
    edges = torch.linspace(min_val, max_val, num_bins, device=x.device)
    bin_width = (max_val - min_val) / num_bins

    # Compute the soft histogram via Gaussian smoothing
    expanded_x = x.unsqueeze(-1)  # Shape (N, 1)
    diff = expanded_x - edges
    soft_assignments = torch.exp(-0.5 * (diff / sigma) ** 2)
    soft_hist = soft_assignments.sum(dim=0)

    return soft_hist / (x.numel() * bin_width)  # Normalize histogram

def mutual_information(moving, static, num_bins=32, normalized=True):
    """
    Differentiable Mutual Information for two images.
    :param moving: torch.Tensor, the moving image.
    :param static: torch.Tensor, the static image.
    :param num_bins: Number of bins for histograms.
    :param normalized: Whether to normalize the mutual information.
    :return: Mutual Information (NMI or MI).
    """
    # Flatten images
    moving_flat = moving.view(-1)
    static_flat = static.view(-1)

    # Compute soft histograms
    moving_hist = soft_histogram(moving_flat, num_bins, moving_flat.min(), moving_flat.max())
    static_hist = soft_histogram(static_flat, num_bins, static_flat.min(), static_flat.max())

    joint_hist = soft_histogram(torch.stack([moving_flat, static_flat], dim=-1).mean(dim=-1),
                                num_bins, min(moving_flat.min(), static_flat.min()),
                                max(moving_flat.max(), static_flat.max()))

    # Compute entropies
    moving_entropy = -torch.sum(moving_hist * torch.log(moving_hist + 1e-10))
    static_entropy = -torch.sum(static_hist * torch.log(static_hist + 1e-10))
    joint_entropy = -torch.sum(joint_hist * torch.log(joint_hist + 1e-10))

    # Mutual Information (MI)
    mi = moving_entropy + static_entropy - joint_entropy

    if normalized:
        # Normalized Mutual Information (NMI)
        return 2.0 * mi / (moving_entropy + static_entropy)
    else:
        return mi
