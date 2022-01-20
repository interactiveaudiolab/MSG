from typing import List
import torch

class AutoBalance(nn.Module):
    def __init__(
        self, ratios: List[float] = [1], frequency: int = 1, max_iters: int = None
    ):
        """
        Auto-balances losses with each other by solving a system of
        equations.
        """
        super().__init__()

        self.frequency = frequency
        self.iters = 0
        self.max_iters = max_iters
        self.weights = [1 for _ in range(len(ratios))]

        # Set up the problem
        ratios = torch.from_numpy(np.array(ratios))

        n_losses = ratios.shape[0]

        off_diagonal = torch.eye(n_losses) - 1
        diagonal = (n_losses - 1) * torch.eye(n_losses)

        A = off_diagonal + diagonal
        B = torch.zeros(1 + n_losses)
        B[-1] = 1

        W = 1 / ratios

        self.register_buffer("A", A.float())
        self.register_buffer("B", B.float())
        self.register_buffer("W", W.float())
        self.ratios = ratios

    def __call__(self, *loss_vals):
        exceeded_iters = False
        if self.max_iters is not None:
            exceeded_iters = self.iters >= self.max_iters

        with torch.no_grad():
            if self.iters % self.frequency == 0 and not exceeded_iters:
                num_losses = self.ratios.shape[-1]
                L = self.W.new(loss_vals[:num_losses])
                if L[L > 0].shape == L.shape:
                    _A = self.A * L * self.W
                    _A = torch.vstack([_A, torch.ones_like(self.W)])

                    # Solve with least squares for weights so each
                    # loss function matches what is given in ratios.
                    X = torch.linalg.lstsq(_A.float(), self.B.float(), rcond=None)[0]

                    self.weights = X.tolist()

        self.iters += 1
        return [w * l for w, l in zip(self.weights, loss_vals)]

def mel_spec_loss(target,estimated):
    eps = 1e-5

    target_spec = torch.stft(input=target, nfft=1024)
    real_part, imag_part = target_spec.unbind(-1)
    target_mag_spec = torch.log10(torch.sqrt(real_part**2 + imag_part**2 + eps))
    
    estimated_spec = torch.stft(input=estimated, nfft=1024)
    real_part, imag_part = estimated_spec.unbind(-1)
    estimated_mag_spec = torch.log10(torch.sqrt(real_part**2 + imag_part**2 +eps))

    return F.l1_loss(target_mag_spec,estimated_mag_spec)