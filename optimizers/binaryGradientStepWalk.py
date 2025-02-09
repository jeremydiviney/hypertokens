import torch
from torch.optim import Optimizer


class BinaryGradientStepWalkOptimizer(Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        lr_decay: float,
        lr_grow: float,
        weight_decay: float = 0,
        lr_bad_epoch_decay: float = 0,
        momentum: int = 0,
        max_step_count: int = 1,
        undo: bool = False,
    ):
        defaults = dict(
            lr=lr,
            lr_decay=lr_decay,
            lr_grow=lr_grow,
            weight_decay=weight_decay,
            lr_bad_epoch_decay=lr_bad_epoch_decay,
            momentum=momentum,
            max_step_count=max_step_count,
            undo=undo,
        )
        super().__init__(params, defaults)

        self.momentums = {}
        for group in self.param_groups:
            for p in group["params"]:
                self.momentums[p] = torch.zeros_like(p.data, dtype=torch.bfloat16)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # mom_factor = self.momentums[p]
                # p.add_(-group["lr"] * mom_factor)
                grad_sign = torch.sign(p.grad)
                p.add_(-group["lr"] * grad_sign)
                # p.add_(-group["lr"] * mom_factor)
                # p.mul_(1 - group["weight_decay"])

        # self.update_momentums()

        return loss

    def update_momentums(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                sign = torch.sign(p.grad)
                # Convert momentum to float16 for computation, then round and convert back.
                momentum = self.momentums[p]
                momentum.add_(sign)
                momentum.mul_(0.925)

    def undo_step(self, step_multiplier=1):
        for i, group in enumerate(self.param_groups):
            for param in group["params"]:
                if param.grad is None:
                    continue

                mom_factor = self.momentums[i].to(torch.float16)
                param.add_(((mom_factor) * group["lr"] * step_multiplier))
                param.div_(1 - group["weight_decay"])

    def lr_decay_step(self):
        for group in self.param_groups:
            group["lr"] *= 1 - group["lr_decay"]

    def lr_increase_step(self):
        for group in self.param_groups:
            group["lr"] += group["lr_grow"]
