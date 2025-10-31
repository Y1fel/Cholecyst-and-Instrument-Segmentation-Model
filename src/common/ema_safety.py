import copy
import numpy as np
import torch

class EMASafetyManager:
    """
    管理EMA模型、异常检测、回滚和冷却机制
    """
    def __init__(self, model, alpha=0.99, loss_window_size=10, grad_explode_thresh=10.0, cooldown_period=5):
        self.ema_alpha = alpha
        self.ema_model = copy.deepcopy(model)
        for p in self.ema_model.parameters():
            p.requires_grad = False
        self.loss_window = []
        self.loss_window_size = loss_window_size
        self.grad_explode_thresh = grad_explode_thresh
        self.cooldown_steps = 0
        self.cooldown_period = cooldown_period

        # === 统计信息 ===
        self.total_steps = 0       # 调用了 step() 的总次数
        self.skipped_updates = 0   # 被跳过的次数（触发了安全机制）

    def update_ema(self, model):
        for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(self.ema_alpha).add_(p.data * (1.0 - self.ema_alpha))

    def copy_ema_to_model(self, model):
        for p, ema_p in zip(model.parameters(), self.ema_model.parameters()):
            p.data.copy_(ema_p.data)

    def check_loss_anomaly(self, loss):
        self.loss_window.append(loss)
        if len(self.loss_window) > self.loss_window_size:
            self.loss_window.pop(0)
        if len(self.loss_window) < self.loss_window_size:
            return False
        mean_loss = np.mean(self.loss_window[:-1])
        if loss > 2.0 * mean_loss:
            return True
        return False

    def check_grad_explosion(self, model):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm > self.grad_explode_thresh

    def step(self, loss, model):
        """
        每步训练后调用。返回True表示本步应跳过参数更新（冷却期），False表示可正常更新。
        """
        # === 统计：每次调用 step 都计数一次 ===
        self.total_steps += 1

        anomaly = self.check_loss_anomaly(loss) or self.check_grad_explosion(model)
        if self.cooldown_steps > 0:
            self.cooldown_steps -= 1
            self.skipped_updates += 1
            return True  # 冷却期，跳过参数更新
        elif anomaly:
            self.copy_ema_to_model(model)
            self.cooldown_steps = self.cooldown_period
            self.skipped_updates += 1
            return True  # 进入冷却期，跳过参数更新
        else:
            self.update_ema(model)
            return False  # 正常更新

    def report(self):
        """打印统计信息"""
        if self.total_steps == 0:
            ratio = 0.0
        else:
            ratio = self.skipped_updates / self.total_steps
        print(f"[EMA Summary] Total checks: {self.total_steps}, "
              f"Skipped updates: {self.skipped_updates}, "
              f"Skip ratio: {ratio:.2%}")
