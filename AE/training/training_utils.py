import torch
import torch.nn as nn
import numpy as np
np.random.seed(42)  

def add_noise(y, denoise_percent):
    """
    Randomly set a percentage of non-zero values to zero for denoising autoencoder.
    
    Parameters:
    -----------
    y : torch.Tensor
        Input tensor (N, seq) where N is number of samples
    denoise_percent : float
        Percentage of non-zero values to set to zero (0.0 to 1.0)
    
    Returns:
    --------
    y_noisy : torch.Tensor
        Noisy version of input with some non-zero values set to zero
    mask : torch.Tensor
        Boolean mask indicating which values were set to zero
    """
    if denoise_percent <= 0.0:
        return y, torch.zeros_like(y, dtype=torch.bool)

    y_noisy = y.clone()
    mask = torch.zeros_like(y, dtype=torch.bool)

    N = y_noisy.size(0)  # Number of samples

    for n in range(N):
        nz = torch.nonzero(y_noisy[n] != 0, as_tuple=True)[0]
        num_non_zero = nz.numel()
        if num_non_zero == 0:
            continue

        num_to_zero = int(num_non_zero * denoise_percent)
        if num_to_zero <= 0:
            continue

        perm = torch.randperm(num_non_zero, device=y_noisy.device)[:num_to_zero]
        seq_idx = nz[perm]

        y_noisy[n, seq_idx] = 0
        mask[n, seq_idx] = True
        # print(y[n, seq_idx], y_noisy[n, seq_idx])
    return y_noisy, mask


# Embed the chromosome feature if needed
class ChromosomeEmbedding(nn.Module):
    def __init__(self, num_chromosomes=17, embedding_dim=4):
        super(ChromosomeEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_chromosomes, embedding_dim)
        
    def forward(self, x):
        return self.embedding(x)


def gaussian_kl(mu, logvar, reduction='mean', normalize_by_dims=None):
    """
    KL divergence between N(mu, sigma^2) and N(0, 1)

    Parameters:
    -----------
    mu : torch.Tensor
        Mean of the latent distribution (batch_size, latent_dim)
    logvar : torch.Tensor
        Log variance of the latent distribution (batch_size, latent_dim)
    reduction : str
        'mean'      : mean over batch (standard VAE loss)
        'sum'       : sum over batch
        'none'      : per-sample KL
    normalize_by_dims : int, optional
        If provided, divide KL by this number (e.g., seq_length) to match
        the scale of reconstruction loss that averages over all output dimensions
    """
    logvar = torch.clamp(logvar, min=-20, max=10)

    kl = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp(),
        dim=1  # sum over latent dims
    )
    
    # Normalize by number of output dimensions to match reconstruction loss scale
    if normalize_by_dims is not None:
        kl = kl / normalize_by_dims

    if reduction == 'mean':
        return kl.mean()
    elif reduction == 'sum':
        return kl.sum()
    elif reduction == 'none':
        return kl
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

def dataloader_from_array(input, chrom=True, batch_size=64, shuffle=True, binary=False, zinb=False, sample_fraction=1.0, denoise_percentage=0.0):
    """
    IMPORTANT: This code currently only performs masking correctly for ZINB mode.
    Create a DataLoader from a numpy array.
    
    Parameters:
    -----------
    sample_fraction : float
        Fraction of data to randomly sample (0.0 to 1.0). Default=1.0 (use all data).
        If < 1.0, will randomly select that fraction of samples once (same subset for all epochs).
    denoise_percentage : float
        Percentage of non-zero values to randomly set to zero for denoising (0.0 to 1.0). Default=0.0
        If > 0, creates fixed masks that are applied once and returned with the data.
    """
    # Load data from file or use array directly
    if isinstance(input, np.ndarray):
        data_array = input
    else:
        data_array = np.load(input)
    
    if denoise_percentage > 0.0 and zinb is False:
        raise NotImplementedError("Denoising masking is only correctly implemented for ZINB mode currently.")
    
    # Random sampling if requested
    if sample_fraction < 1.0:
        num_samples = data_array.shape[0]
        num_to_sample = int(num_samples * sample_fraction)
        sampled_indices = np.random.choice(num_samples, size=num_to_sample, replace=False)
        # sampled_indices = np.sort(sampled_indices)  # Sort to maintain some order
        data_array = data_array[sampled_indices]
        print(f"Randomly sampled {num_to_sample}/{num_samples} samples ({sample_fraction*100:.1f}%)")
    
    counts = data_array[:, :, 0]  # Normalized counts (Value)
    
    # Apply denoising masking once if requested
    if denoise_percentage > 0.0:
        counts_noisy, mask = add_noise(torch.tensor(counts, dtype=torch.float32), denoise_percentage)
        counts = counts_noisy.numpy()  # Replace counts with masked version
        mask_array = mask.numpy()  # Convert mask to numpy for later tensor conversion
        print(f"Applied denoising with {denoise_percentage*100:.1f}% masking")
    else:
        mask_array = np.zeros_like(counts, dtype=bool)  # No masking

    
    # For ZINB mode: extract Value_Raw and Size_Factor from the end of the array
    if chrom:
        # Structure: [Value, features..., Chrom, Value_Raw, Size_Factor]
        features = data_array[:, :, 1:-3]  # Exclude Value, Chrom, Value_Raw, Size_Factor
        chrom_indices = data_array[:, :, -3].astype(np.int64)
        raw_counts = data_array[:, :, -2]  # Value_Raw
        size_factors = data_array[:, 0, -1]  # Size_Factor (take first position, all same)
    else:
        # Structure: [Value, features..., Value_Raw, Size_Factor]
        features = data_array[:, :, 1:-2]  # Exclude Value, Value_Raw, Size_Factor
        raw_counts = data_array[:, :, -2]  # Value_Raw
        size_factors = data_array[:, 0, -1]  # Size_Factor (take first position, all same)
    
    x_tensor = torch.tensor(features, dtype=torch.float32)
    y_tensor = torch.tensor(counts, dtype=torch.float32)  # This is y_noisy if denoise_percentage > 0
    mask_tensor = torch.tensor(mask_array, dtype=torch.bool)
    
    
    # Build dataset tensors
    tensors = [x_tensor, y_tensor]
    
    if chrom:
        c_tensor = torch.tensor(chrom_indices, dtype=torch.long)
        tensors.append(c_tensor)
    
    y_raw_tensor = torch.tensor(raw_counts, dtype=torch.float32)  # Always unmasked (original)
    sf_tensor = torch.tensor(size_factors, dtype=torch.float32)
    tensors.append(y_raw_tensor)
    tensors.append(sf_tensor)
    
    # Always append mask tensor (even if all False when denoise_percentage=0)
    tensors.append(mask_tensor)
    
    dataset = torch.utils.data.TensorDataset(*tensors)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader