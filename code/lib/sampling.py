import sklearn
import torch
import numpy as np
import torch.nn.functional as F

@torch.no_grad()
def sample(model, diffusion, cond, num_features, mask=None, device='cuda'):

    """
    Reverse diffusion sampling.
    """
    
    #  check if it's a list
    if isinstance(cond, list):
        cond = [c.to(device) for c in cond]
        num_samples = cond[0].size(0)
    else:
        cond = cond.to(device)
        num_samples = cond.size(0)

    # Handle mask similarly
    if mask is not None:   
        if isinstance(mask, list):
            mask = [m.to(device) for m in mask]
        else:
            mask = mask.to(device)
    
    # Initialize samples with random noise
    samples = torch.randn(num_samples, num_features, device=device)


    for t in reversed(range(diffusion.num_timesteps)):
        t_tensor = torch.tensor([t] * num_samples, dtype=torch.long, device=device)
        predicted_noise = model(samples, t_tensor, cond, mask)
        
        beta_t = diffusion.get_beta(t)
        alpha_bar_t = diffusion.get_alpha_bar(t)
        
        if t > 0:
            alpha_bar_prev = diffusion.get_alpha_bar(t - 1)
            alpha_t = alpha_bar_t / alpha_bar_prev
            sigma_t = torch.sqrt(beta_t)
        else:
            alpha_t = alpha_bar_t
            sigma_t = 0.0

        # Reverse diffusion update:
        samples = (samples - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * predicted_noise) / torch.sqrt(alpha_t)
        
        if t > 0:
            noise = torch.randn_like(samples)
            samples += sigma_t * noise

    return samples

@torch.no_grad()
def coherent_sample(models, diffusion, num_samples, num_features, conds, device='cuda', return_cos=False, weights=None, masks=None):
    """
    Generates new samples using coherent sampling — fully on GPU.
    """

    samples = torch.randn((num_samples, num_features), dtype=torch.float32, device=device)

    # Handle weights (optional MSE-based weighting)
    if weights is not None:
        weights = F.softmax(torch.tensor([1.0 / w for w in weights], device=device), dim=0)


    cos_per_timestep = []

    for t in reversed(range(diffusion.num_timesteps)):
        t_tensor = torch.full((num_samples,), t, dtype=torch.long, device=device)

        # Get the predicted noise from each model
        if masks is not None:
            predicted_noises = torch.stack([model(samples, t_tensor, cond, mask) for model, cond, mask in zip(models, conds, masks)]) 
        else:
            predicted_noises = torch.stack([model(samples, t_tensor, cond) for model, cond in zip(models, conds)]) 
            

        #  Compute, for each sample, the average pairwise cosine distance among the predicted noises
        #    - Let M = number of models
        #    - For each pair of distinct models (i < j), compute the cosine similarity between
        #      predicted_noises[i, n, :] and predicted_noises[j, n, :] for each sample n.
        #    - Average those similarities over all (i,j) pairs, then do (1 - avg_similarity)
        #      to get the average cosine distance. That yields a vector of length num_samples.
        M = predicted_noises.shape[0]
        N = num_samples

        sim_acc = torch.zeros((N,), device=device)

        if weights is None or M == 1:
            # unweighted average over pairs
            num_pairs = M * (M - 1) / 2 if M > 1 else 1.0
            for i in range(M):
                for j in range(i + 1, M):
                    sim_acc += F.cosine_similarity(predicted_noises[i], predicted_noises[j], dim=1)
            avg_sim = sim_acc / num_pairs
        else:
            # additive mean weighting: w_ij = (w_i + w_j) / 2
            sum_pair_w = 0.0
            for i in range(M):
                for j in range(i + 1, M):
                    w_ij = 0.5 * (weights[i] + weights[j])
                    sum_pair_w += w_ij
                    sim_acc += w_ij * F.cosine_similarity(predicted_noises[i], predicted_noises[j], dim=1)
            avg_sim = sim_acc / sum_pair_w

        avg_cos_dist = 1.0 - avg_sim  # shape: (N,)
        cos_per_timestep.append(avg_cos_dist)

        # Weighted or unweighted combination
        if weights is None:
            predicted_noise = predicted_noises.mean(dim=0)
        else:
            predicted_noise = torch.sum(predicted_noises * weights.view(-1, 1, 1), dim=0)

        # Get diffusion parameters as tensors
        beta_t = diffusion.get_beta(t).to(device)
        alpha_bar_t = diffusion.get_alpha_bar(t)
        alpha_bar_t = torch.tensor(alpha_bar_t, device=device) if not isinstance(alpha_bar_t, torch.Tensor) else alpha_bar_t

        if t > 0:
            alpha_bar_prev = diffusion.get_alpha_bar(t - 1)
            alpha_bar_prev = torch.tensor(alpha_bar_prev, device=device) if not isinstance(alpha_bar_prev, torch.Tensor) else alpha_bar_prev
            alpha_t = alpha_bar_t / alpha_bar_prev
            sigma_t = torch.sqrt(beta_t)
        else:
            alpha_t = torch.tensor(1.0, device=device)
            sigma_t = torch.tensor(0.0, device=device)

        # Update the samples using the predicted noise
        samples = (samples - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise) / torch.sqrt(alpha_t)

        if t > 0:
            noise = torch.randn_like(samples)
            samples += sigma_t * noise



    if return_cos:
        # Stack along new dimension=1 to get shape (T, num_samples), then transpose → (num_samples, T)
        cos_stack = torch.stack(cos_per_timestep, dim=0)  # shape: (T, num_samples)
        cos_trajectory = cos_stack.transpose(0, 1).contiguous()  # shape: (num_samples, T)
        return samples, cos_trajectory
    
    else:
        return samples