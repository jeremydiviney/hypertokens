import torch
from torch.optim.optimizer import Optimizer


class SignSGD(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9):
        defaults = dict(lr=lr, momentum=momentum)
        super(SignSGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                # Initialize or retrieve momentum buffer
                state = self.state[p]
                if "momentum_buffer" not in state:
                    buf = state["momentum_buffer"] = torch.zeros_like(p.data)
                else:
                    buf = state["momentum_buffer"]

                # Update buffer and apply sign-based step
                buf.mul_(momentum).add_(grad, alpha=1 - momentum)
                p.data -= lr * buf.sign()
