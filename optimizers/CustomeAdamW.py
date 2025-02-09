import torch
from torch.optim.optimizer import Optimizer


class CustomAdamW(Optimizer):
    r"""Implements AdamW algorithm.

    Arguments:
        params (iterable): Iterable of parameters to optimize.
        lr (float, optional): Learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): Coefficients used for computing
            running averages of gradient and its square. (default: (0.9, 0.999))
        eps (float, optional): Term added to the denominator for numerical stability.
            (default: 1e-8)
        weight_decay (float, optional): Decoupled weight decay (L2 penalty). (default: 0.01)
        amsgrad (bool, optional): Whether to use the AMSGrad variant of this algorithm.
            (default: False)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super(CustomAdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            amsgrad = group["amsgrad"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients.")

                state = self.state[p]

                # State initialization.
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]

                state["step"] += 1
                step = state["step"]

                # Decoupled weight decay.
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)

                # Update biased first moment estimate.
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Update biased second raw moment estimate.
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if amsgrad:
                    # Maintains maximum of all 2nd moment running avg. till now.
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(eps)
                else:
                    denom = exp_avg_sq.sqrt().add_(eps)

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                step_size = lr * (bias_correction2**0.5) / bias_correction1

                # Update parameters.
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
