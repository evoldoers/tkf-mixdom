"""Early stopping with validation-based patience.

Tracks validation LL across iterations, stops when no improvement
for `patience` consecutive evaluations.

Usage:
    stopper = EarlyStopper(patience=5)

    for iter in range(max_iters):
        train_one_iter(...)
        if iter % val_every == 0:
            val_ll = evaluate_on_val(...)
            if stopper.step(val_ll, params):
                print(f"Early stopping at iter {iter}")
                best_params = stopper.best_params
                break
"""

import copy
import numpy as np


class EarlyStopper:
    """Patience-based early stopping on validation metric.

    Tracks the best validation LL seen so far. If `patience`
    consecutive evaluations fail to improve, signals to stop.
    Saves the best params checkpoint for recovery.
    """

    def __init__(self, patience=5, min_delta=0.0):
        """
        Args:
            patience: number of val evals without improvement before stopping
            min_delta: minimum improvement to count as "better"
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_ll = -np.inf
        self.best_params = None
        self.best_iter = -1
        self.wait = 0
        self.history = []  # (iter, val_ll, is_best)

    def step(self, val_ll, params, iter_num=0):
        """Record a validation result. Returns True if should stop.

        Args:
            val_ll: validation log-likelihood (higher = better)
            params: current parameter dict (deep-copied if best)
            iter_num: current iteration number (for logging)

        Returns:
            True if patience exhausted (should stop training)
        """
        is_best = val_ll > self.best_ll + self.min_delta
        self.history.append((iter_num, val_ll, is_best))

        if is_best:
            self.best_ll = val_ll
            self.best_iter = iter_num
            self.best_params = _deep_copy_params(params)
            self.wait = 0
        else:
            self.wait += 1

        return self.wait >= self.patience

    def report(self):
        """Return a summary string."""
        return (f"best_ll={self.best_ll:.2f} at iter {self.best_iter}, "
                f"wait={self.wait}/{self.patience}")


def _deep_copy_params(params):
    """Deep copy a params dict (numpy arrays)."""
    out = {}
    for k, v in params.items():
        if isinstance(v, np.ndarray):
            out[k] = v.copy()
        else:
            out[k] = v
    return out
