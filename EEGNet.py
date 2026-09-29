import torch
import torch.nn as nn
import numpy as np

class EEGNet(nn.Module):
    def __init__(self, n_classes=4, n_channels=22, n_samples=500,
                 dropout_rate=0.5, kernel_length=64, F1=8, D=2, F2=16):
        super().__init__()

        # Block 1: temporal conv + depthwise spatial conv
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),  # depthwise
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate)
        )

        # Block 2: separable conv
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),  # depthwise
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),  # pointwise
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate)
        )

        # Compute flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            out = self.block2(self.block1(dummy))
            flat_dim = out.numel()

        self.classify = nn.Linear(flat_dim, n_classes)

    def forward(self, x):
        # x: (batch, 1, n_channels, n_samples)
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(start_dim=1)
        return self.classify(x)


# Training functions

def _train_round(model, loader, optimizer, criterion, device, metrics, augmentation_fns=None, aug_prob=0.5):
    model.train()

    train_loss = np.zeros(len(loader))
    y_pred_all, y_true_all = [], []

    for idx_batch, (batch_x, batch_y) in enumerate(loader):
        optimizer.zero_grad()

        batch_x = batch_x.to(device=device, dtype=torch.float32)
        batch_y = batch_y.to(device=device, dtype=torch.long)

        if augmentation_fns != None and np.random.rand() < aug_prob:
            aug_fn = np.random.choice(augmentation_fns)
            batch_x, batch_y = aug_fn(batch_x, batch_y)

        output = model(batch_x)
        loss = criterion(output, batch_y)

        loss.backward()
        optimizer.step()

        y_pred_all.append(torch.argmax(output, axis=1).cpu().numpy())
        y_true_all.append(batch_y.cpu().numpy())

        train_loss[idx_batch] = loss.item()

    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)

    perf = []
    for metric in metrics:
        perf.append(metric(y_true, y_pred))

    return np.mean(train_loss), perf

def _validate(model, loader, criterion, device, metrics):
    model.eval()

    val_loss = np.zeros(len(loader))
    y_pred_all, y_true_all = [], []

    with torch.no_grad():
        for idx_batch, (batch_x, batch_y) in enumerate(loader):
            batch_x = batch_x.to(device=device, dtype=torch.float32)
            batch_y = batch_y.to(device=device, dtype=torch.long)

            output = model.forward(batch_x)

            loss = criterion(output, batch_y)
            val_loss[idx_batch] = loss.item()

            y_pred_all.append(torch.argmax(output, axis=1).cpu().numpy())
            y_true_all.append(batch_y.cpu().numpy())

    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)

    perf = []
    for metric in metrics:
        perf.append(metric(y_true, y_pred))

    return np.mean(val_loss), perf

import copy

def train(model, loader_train, loader_valid, optimizer, criterion, n_epochs, patience, device, metrics, augmentation_fns=None, aug_prob=0.5):

    best_valid_loss = np.inf
    best_model = copy.deepcopy(model)
    waiting = 0
    history = []

    for epoch in range(1, n_epochs + 1):
        train_loss, train_perf = _train_round(model, loader_train, optimizer, criterion, device, metrics=metrics, augmentation_fns=augmentation_fns, aug_prob=aug_prob)
        valid_loss, valid_perf = _validate(model, loader_valid, criterion, device, metrics=metrics)
        # train_loss, valid_loss, train_perf, valid_perf = 1.9462, 1.9392, 0.0744, 0.0259

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'train_perf': train_perf,
            'valid_perf': valid_perf
        })

        print(f'{epoch} \t {train_loss:0.4f} \t {valid_loss:0.4f} 'f'\t {train_perf[0]:0.4f} \t {valid_perf[0]:0.4f}')

        # model saving
        if valid_loss < best_valid_loss:
            print(f'best val loss {best_valid_loss:.4f} -> {valid_loss:.4f}')
            best_valid_loss = valid_loss
            best_model = copy.deepcopy(model)
            waiting = 0
        else:
            waiting += 1


        # model early stopping
        if waiting >= patience:
            print(f'Stop training at epoch {epoch}')
            print(f'Best val loss : {best_valid_loss:.4f}')
            break

    return best_model, history