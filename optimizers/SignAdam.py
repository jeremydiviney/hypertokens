import torch
from torch.optim.optimizer import Optimizer


class SignAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(SignAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                # Update running averages
                state["step"] += 1
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias-corrected first moment (m_hat)
                bias_correction1 = 1 - beta1 ** state["step"]
                m_hat = exp_avg / bias_correction1

                # Sign-based parameter update
                p.data.add_(-lr, m_hat.sign())


import torch
from torch.optim.optimizer import Optimizer


class MixedSignAdam(Optimizer):
    """
    MixedSignAdamStable computes updates as:
       update = [ (1 - gamma) * m̂ + gamma * sign(m̂) ] / (clamp(sqrt(v̂), min_denom) + eps)
    where m̂ and v̂ are bias-corrected first and second moments.

    gamma = 0 → pure Adam update.
    gamma = 1 → pure sign-based update scaled by the second moment.

    Clamping the denominator helps avoid enormous updates when v̂ is very small.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-4, gamma=0.9, min_denom=1e-3):
        defaults = dict(lr=lr, betas=betas, eps=eps, gamma=gamma, min_denom=min_denom)
        super(MixedSignAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            gamma = group["gamma"]
            min_denom = group["min_denom"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # State initialization.
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                # Update biased first and second moment estimates.
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias-corrected estimates.
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2

                # Blend full m_hat with its sign.
                combined = (1 - gamma) * m_hat + gamma * m_hat.sign()

                # Clamp sqrt(v_hat) to avoid division by a too-small value.
                denom = v_hat.sqrt().clamp(min=min_denom)
                update = combined / (denom + eps)

                p.add_(update, alpha=-lr)
