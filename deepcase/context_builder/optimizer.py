import torch.optim as optim

class VarAdam(optim.Adam):
    """Adam optimizer with variable learning rate."""

    def __init__(self, model, factor=1, warmup=4000, optimizer=optim.Adam, lr=0, betas=(0.9, 0.98), eps=1e-9):
        # Initialise super
        super().__init__(
            params = model.parameters(),
            lr     = lr,
            betas  = betas,
            eps    = eps,
        )

        # Set variables
        self.warmup    = warmup
        self.factor    = factor
        self.dimension = model.output_size
        self._step     = 0
        self._rate     = 0

    def step(self):
        """Update parameters and rate"""
        # Increment step
        self._step += 1
        # Get learning rate
        self._rate = self.rate()
        # Set learning rates
        for parameter in self.param_groups:
            parameter['lr'] = self._rate
        # Set optimizer step
        super().step()

    def rate(self, step=None):
        """Compute current learning rate

            Parameters
            ----------
            step : int, (optional)
                Number of steps to take
            """
        # Get current step if None given
        if step is None: step = self._step
        # Compute learning rate
        return self.factor            *\
               self.dimension**(-0.5) *\
               min(step**(-0.5), step*self.warmup**(-1.5))
