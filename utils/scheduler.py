# ---------------------------------------------------------------
# Original code from https://github.com/pytorch/pytorch/blob/main/torch/optim/lr_scheduler.py.
# ---------------------------------------------------------------


import torch
import warnings
import math


class PolynomialLR(torch.optim.lr_scheduler._LRScheduler):
    """Decays the learning rate of each parameter group using a polynomial function
    in the given total_iters. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        total_iters (int): The number of steps that the scheduler decays the learning rate. Default: 5.
        power (float): The power of the polynomial. Default: 1.0.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """
    def __init__(self, optimizer, total_iters=5, power=1.0, last_epoch=-1, verbose=False):
        self.total_iters = total_iters
        self.power = power
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0 or self.last_epoch > self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]

        decay_factor = ((1.0 - self.last_epoch / self.total_iters) / (1.0 - (self.last_epoch - 1) / self.total_iters)) ** self.power
        return [group["lr"] * decay_factor for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [
            (
                base_lr * (1.0 - min(self.total_iters, self.last_epoch) / self.total_iters) ** self.power
            )
            for base_lr in self.base_lrs
        ]

class WarmupMultiStepLR(torch.optim.lr_scheduler.MultiStepLR):
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=1.0 / 3,
                 warmup_iters=500, last_epoch=-1):
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        super().__init__(optimizer, milestones, gamma, last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.warmup_iters:
            alpha = self.last_epoch / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            # print(self.base_lrs[0]*warmup_factor)
            return [lr * warmup_factor for lr in self.base_lrs]
        else:
            lr = super().get_lr()
        return lr
    
class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_max, warmup_factor=1.0 / 3, warmup_iters=500,
                 eta_min=0, last_epoch=-1):
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.T_max, self.eta_min = T_max, eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.warmup_iters:
            alpha = self.last_epoch / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            # print(self.base_lrs[0]*warmup_factor)
            return [lr * warmup_factor for lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(
                        math.pi * (self.last_epoch - self.warmup_iters) / (self.T_max - self.warmup_iters))) / 2
                    for base_lr in self.base_lrs]
        


class WarmupPolyLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_max, cur_iter, warmup_factor=1.0 / 3, warmup_iters=500,
                 eta_min=0, power=0.9):
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.power = power
        self.T_max, self.eta_min = T_max, eta_min
        self.cur_iter = cur_iter
        super().__init__(optimizer)

    def get_lr(self):
        if self.cur_iter <= self.warmup_iters:
            alpha = self.cur_iter / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            # print(self.base_lrs[0]*warmup_factor)
            # print(f"cur_iter: {self.cur_iter}, warmup_iters: {self.warmup_iters}, warmup_factor: {warmup_factor}")
            # print(f"base_lrs: {self.base_lrs}")
            # print(f"lr: {[lr * warmup_factor for lr in self.base_lrs]}")
            return [lr * warmup_factor for lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr - self.eta_min) *
                    math.pow(1 - (self.cur_iter - self.warmup_iters) / (self.T_max - self.warmup_iters),
                             self.power) for base_lr in self.base_lrs]


def poly_learning_rate(cur_epoch, max_epoch, curEpoch_iter, perEpoch_iter, baselr):
    cur_iter = cur_epoch * perEpoch_iter + curEpoch_iter
    max_iter = max_epoch * perEpoch_iter
    lr = baselr * pow((1 - 1.0 * cur_iter / max_iter), 0.9)

    return lr



class GradualWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        min_lr_mul: target learning rate = base lr * min_lr_mul
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, total_epoch, min_lr_mul=0.1, after_scheduler=None):
        self.min_lr_mul = min_lr_mul
        if self.min_lr_mul > 1. or self.min_lr_mul < 0.:
            raise ValueError('min_lr_mul should be [0., 1.]')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = self.base_lrs
                    self.finished = True
                return self.after_scheduler.get_lr()
            else:
                return self.base_lrs
        else:
            return [base_lr * (self.min_lr_mul + (1. - self.min_lr_mul) * (self.last_epoch / float(self.total_epoch))) for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            return self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)