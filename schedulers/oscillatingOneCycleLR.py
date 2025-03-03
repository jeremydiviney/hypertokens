import math
from torch.optim.lr_scheduler import _LRScheduler


class OscillatingOneCycleLR(_LRScheduler):
    """
    Scheduler that combines OneCycleLR's overall shape with cosine oscillations.

    This scheduler creates a learning rate that follows a one-cycle policy with a
    cosine annealing pattern, but with additional smaller cosine oscillations on top.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        min_lr (float or list): Lower learning rate boundaries in the cycle.
        max_lr (float or list): Upper learning rate boundaries in the cycle.
        total_steps (int): The total number of steps in the training process.
        pct_start (float): The percentage of the cycle spent increasing the learning rate.
        oscillation_period (int): Period of oscillations in steps (how many steps it takes to complete one oscillation).
        oscillation_amplitude (float): Amplitude of oscillations as a fraction of the current LR (0.0 to 1.0).
        anneal_strategy (str): {'cos', 'linear'} Specifies the annealing strategy for the larger cycle.
    """

    def __init__(
        self,
        optimizer,
        min_lr,
        max_lr,
        total_steps,
        pct_start=0.1,
        oscillation_period=100,
        oscillation_amplitude=0.2,
        anneal_strategy="cos",
    ):

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.oscillation_period = oscillation_period
        self.oscillation_amplitude = oscillation_amplitude
        self.anneal_strategy = anneal_strategy
        self.warmup_steps = 0

        # Validate inputs
        if oscillation_amplitude < 0.0:
            raise ValueError(f"oscillation_amplitude must be greater than 0.0, got {oscillation_amplitude}")

        if anneal_strategy not in ["cos", "linear"]:
            raise ValueError(f"anneal_strategy must be 'cos' or 'linear', got {anneal_strategy}")

        # Initialize step count
        self.step_count = 0

        super(OscillatingOneCycleLR, self).__init__(optimizer)

    def get_lr(self):
        """Calculate the learning rate at current step."""
        # Calculate the percentage of the cycle
        if self.step_count >= self.total_steps:
            return [group["lr"] for group in self.optimizer.param_groups]

        cycle_progress = float(self.step_count) / float(self.total_steps)

        lrs = []

        # Split the cycle into two phases: warmup and annealing
        if cycle_progress < self.pct_start:
            # Phase 1: Warmup (linear increase from base_lr to max_lr)
            base_progress = cycle_progress / self.pct_start
            scale_factor = base_progress
            self.warmup_steps += 1

            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                lr = max(self.max_lr * scale_factor, self.min_lr)
                lrs.append(lr)

            return lrs

        else:
            # Phase 2: Annealing (cos or linear decrease from max_lr to min_lr)
            annealing_step_count = self.step_count - self.warmup_steps
            total_annealing_steps = self.total_steps - self.warmup_steps
            annealing_progress = float(annealing_step_count) / float(total_annealing_steps)

            if self.anneal_strategy == "cos":
                scale_factor = 1 - 0.5 * (1 + math.cos(math.pi * (1 - annealing_progress)))
            else:  # 'linear'
                scale_factor = 1 - annealing_progress

            # Add oscillations on top of the base curve
            osc_phase = 2 * math.pi * ((annealing_step_count - 1) / self.oscillation_period)
            osc_term = self.oscillation_amplitude * ((0.5 + math.sin(osc_phase - math.pi / 2) / 2))

            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                # Annealing phase: max_lr to min_lr
                lr = self.max_lr * scale_factor
                lr = lr + lr * osc_term
                lrs.append(lr)
                # print(f"annealing step {annealing_step_count} lr: {lr}, scale_factor: {scale_factor}, osc_term: {osc_term}")

        return lrs

    def step(self, epoch=None):
        """Take a step and update the learning rate."""
        self.step_count += 1
        return super(OscillatingOneCycleLR, self).step(epoch)
