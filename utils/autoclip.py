import numpy as np
import torch

class AutoClip:
    def __init__(self, percentile: float = 10, frequency: int = 1, mask_nan: int = 0):
        """
        Adds AutoClip during training.
        The gradient is clipped to the percentile'th percentile of
        gradients seen during training. Proposed in [1].

        [1] Prem Seetharaman, Gordon Wichern, Bryan Pardo,
            Jonathan Le Roux. "AutoClip: Adaptive Gradient
            Clipping for Source Separation Networks." 2020
            IEEE 30th International Workshop on Machine
            Learning for Signal Processing (MLSP). IEEE, 2020.

        Parameters
        ----------
        percentile : float, optional
            Percentile to clip gradients to, by default 10
        frequency : int, optional
            How often to re-compute the clipping value.
        """
        self.grad_history = []
        self.percentile = percentile
        self.frequency = frequency
        self.mask_nan = bool(mask_nan)

        self.iters = 0

    def state_dict(self):
        state_dict = {
            "grad_history": self.grad_history,
            "percentile": self.percentile,
            "frequency": self.frequency,
            "mask_nan": self.mask_nan,
            "iters": self.iters,
        }
        return state_dict

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            setattr(self, k, v)

    def __call__(self, model):
        if self.iters % self.frequency == 0:
            grad_norm = compute_grad_norm(model, self.mask_nan)
            self.grad_history.append(grad_norm)
        else:
            grad_norm = self.grad_history[-1]

        clip_value = np.percentile(self.grad_history, self.percentile)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        self.iters += 1
        return clip_value, grad_norm

def compute_grad_norm(self, mask_nan=False):
    all_norms = []
    for p in self.parameters():
        if p.grad is None:
            continue
        grad_data = p.grad.data

        if mask_nan:
            nan_mask = torch.isfinite(p.grad.data)
            grad_data = grad_data[nan_mask]
        param_norm = float(grad_data.norm(2))
        all_norms.append(param_norm)

    total_norm = float(torch.tensor(all_norms).norm(2))
    return total_norm