import torch

def si_snr(target, estimate, eps=1e-8):
    """
    Compute Scale-Invariant Signal-to-Noise Ratio (SI-SNR).
    
    Args:
        target (torch.Tensor): Clean target signal (batch_size, time).
        estimate (torch.Tensor): Estimated signal (batch_size, time).
        eps (float): Small value to avoid division by zero.
        
    Returns:
        torch.Tensor: SI-SNR value for each example in the batch.
    """
    # Ensure the inputs have zero mean
    target_mean = torch.mean(target, dim=1, keepdim=True)
    estimate_mean = torch.mean(estimate, dim=1, keepdim=True)
    target = target - target_mean
    estimate = estimate - estimate_mean

    # Compute target projection
    dot_product = torch.sum(estimate * target, dim=1, keepdim=True)
    target_energy = torch.sum(target ** 2, dim=1, keepdim=True) + eps
    target_projection = (dot_product / target_energy) * target

    # Compute noise (residual)
    noise = estimate - target_projection

    # Compute SI-SNR
    si_snr_value = 10 * torch.log10(
        (torch.sum(target_projection ** 2, dim=1) + eps) /
        (torch.sum(noise ** 2, dim=1) + eps)
    )
    return si_snr_value

class SISNRLoss(torch.nn.Module):
    def __init__(self):
        super(SISNRLoss, self).__init__()
    
    def forward(self, target, estimate):
        # Negate SI-SNR to use as a loss
        return -torch.mean(si_snr(target, estimate))
