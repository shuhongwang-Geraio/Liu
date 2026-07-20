"""
训练器：支持PatchTST和CausalCIT的统一训练流程
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from utils.metrics import metric


class EarlyStopping:
    def __init__(self, patience=7, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        torch.save(model.state_dict(), os.path.join(path, 'checkpoint.pth'))
        self.val_loss_min = val_loss


class Trainer:
    """统一训练器"""
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device

    def train(self, train_loader, val_loader, epochs=20, lr=0.001,
              patience=5, save_dir='./checkpoints', entropy_weight=0.0):
        """
        entropy_weight: P1优化 — 门控熵正则化系数 (默认0，不影响旧行为)。
            若>0且模型实现了get_gate_entropy()，会在loss中加入
            `entropy_weight * gate_entropy`，鼓励门控sigmoid输出远离0.5，
            做出更果断的通道交互/隔离判断，避免门控长期停留在模糊区间。
        """
        os.makedirs(save_dir, exist_ok=True)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr,
            steps_per_epoch=len(train_loader), epochs=epochs, pct_start=0.3
        )
        early_stopping = EarlyStopping(patience=patience)
        supports_entropy = entropy_weight > 0 and hasattr(self.model, 'get_gate_entropy')

        train_losses, val_losses = [], []
        start_time = time.time()

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = []
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                if supports_entropy:
                    gate_entropy = self.model.get_gate_entropy()
                    if gate_entropy is not None:
                        loss = loss + entropy_weight * gate_entropy
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss.append(loss.item())

            train_loss = np.mean(epoch_loss)
            val_loss = self.validate(val_loader, criterion)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

            early_stopping(val_loss, self.model, save_dir)
            if early_stopping.early_stop:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        total_time = time.time() - start_time
        # 加载最优模型
        self.model.load_state_dict(torch.load(
            os.path.join(save_dir, 'checkpoint.pth'),
            map_location=self.device, weights_only=True
        ))
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': early_stopping.val_loss_min,
            'total_time': total_time,
            'epochs_trained': len(train_losses),
        }

    def validate(self, val_loader, criterion):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                losses.append(loss.item())
        return np.mean(losses)

    def test(self, test_loader):
        self.model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                pred = self.model(batch_x)
                preds.append(pred.cpu().numpy())
                trues.append(batch_y.cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        mse, mae, rmse, rse, corr = metric(preds, trues)
        return {
            'mse': mse, 'mae': mae, 'rmse': rmse, 'rse': rse, 'corr': corr,
            'preds': preds, 'trues': trues,
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
