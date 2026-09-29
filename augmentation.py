import torch
import numpy as np

def add_gaussian_noise(X, y, sigma=0.05):
    noise = torch.randn_like(X) * sigma
    return X + noise, y

def time_shift(X, y, max_shift=25):
    shift = np.random.randint(-max_shift, max_shift)
    return torch.roll(X, shifts=shift, dims=-1), y

def channel_dropout(X, y, p=0.1):
    mask = (torch.rand(X.shape[2], 1) > p).float()  # per-channel mask
    return X * mask.to(X.device), y

def same_class_mixup(X, y, alpha=0.2):
    X_aug, y_aug = [], []
    for cls in torch.unique(y):
        idx = (y == cls).nonzero(as_tuple=True)[0]
        if len(idx) < 2:
            continue
        i, j = idx[torch.randperm(len(idx))[:2]]
        lam = np.random.beta(alpha, alpha)
        mixed = lam * X[i] + (1 - lam) * X[j]
        X_aug.append(mixed)
        y_aug.append(cls)
    return torch.stack(X_aug), torch.tensor(y_aug)

def freq_perturb(X, y, scale=0.1):
    X_np = X.cpu().numpy()
    fft = np.fft.rfft(X_np, axis=-1)
    perturbation = 1 + np.random.uniform(-scale, scale, size=fft.shape)
    fft_perturbed = fft * perturbation
    X_aug = np.fft.irfft(fft_perturbed, n=X_np.shape[-1], axis=-1)
    return torch.tensor(X_aug, dtype=torch.float32).to(X.device), y