from random import shuffle


class EmpiricalLRScheduler:
    """
    Cycle through 3 candidate LRs each for (n//3) batches:
      1) base_lr * (1 + alpha)
      2) base_lr * (1 - alpha)
      3) base_lr
    Track average loss in each chunk, pick the best LR for the next cycle.
    """

    def __init__(self, optimizer, alpha=0.1, n=30):
        self.optimizer = optimizer
        self.alpha = alpha
        self.n = n
        self.chunk_size = n // 3
        self.last_loss = None

        self.base_lr = self.optimizer.param_groups[0]["lr"]

        # Keep track of losses and which chunk we are on
        self.chunk_losses = [0.0, 0.0, 0.0]
        self.step_count = 0

        # Candidate LRs
        self.candidate_lrs = [0.0, 0.0, 0.0]

        # Start with the first candidate
        self._set_lr(self.base_lr)

    def step(self, loss):
        """
        Call this once per batch, passing in the current batch's loss (scalar).
        """
        chunk_idx = self.step_count % 3

        if self.step_count == 0:
            current_lr = self.base_lr
            self.candidate_lrs = [
                current_lr * (1 + self.alpha),
                current_lr * (1 - self.alpha),
                current_lr,
            ]
            shuffle(self.candidate_lrs)

        self.chunk_losses[chunk_idx] += loss

        next_chunk = (chunk_idx + 1) % 3 if chunk_idx + 1 < self.n else 0
        self._set_lr(self.candidate_lrs[next_chunk])

        self.step_count += 1
        self.last_loss = loss

        # If we've finished all 3 chunks in this cycle, pick the best and reset
        if self.step_count == self.n:
            avg_losses = [self.chunk_losses[i] / self.chunk_size for i in range(3)]
            best_idx = min(range(3), key=lambda i: avg_losses[i])
            # Update base_lr to best performing chunk
            self.base_lr = self.candidate_lrs[best_idx]
            self._set_lr(self.base_lr)
            self.step_count = 0
            self.chunk_losses = [0.0, 0.0, 0.0]

    def _set_lr(self, lr_value):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr_value
