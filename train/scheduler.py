import numpy as np


def assign_learning_rate(optimizer, new_lr, para_gamma=0.01):
   for param_group in optimizer.param_groups:
        if len(param_group['params']) == 1:
            param_group["lr"] = new_lr
        else:
            param_group["lr"] = new_lr * para_gamma


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lr, warmup_length, steps, para_gamma=0.01):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr, para_gamma)
        return lr
    return _lr_adjuster
