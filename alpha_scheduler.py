import math


class AlphaScheduler:
    def __init__(
            self,
            initial_alpha=1.0000001,
            final_alpha=2.0,
            max_steps=10000,
            strategy="linear",
            power=2,  # For polynomial annealing
            step_size=1000,  # For stepwise annealing
            increment=0.1,  # For stepwise annealing
            k=0.1  # For sigmoid annealing
    ):
        self.initial_alpha = initial_alpha
        self.final_alpha = final_alpha
        self.max_steps = max_steps
        self.current_step = 0
        self.strategy = strategy
        self.power = power
        self.step_size = step_size
        self.increment = increment
        self.alpha = initial_alpha
        self.k = k

    def step(self):
        self.current_step += 1
        progress = self.current_step / self.max_steps

        if self.strategy == "linear":
            # Linear annealing
            new_alpha = self.initial_alpha + (self.final_alpha - self.initial_alpha) * progress

        elif self.strategy == "exponential":
            # Exponential annealing
            new_alpha = self.initial_alpha * (self.final_alpha / self.initial_alpha) ** progress

        elif self.strategy == "cosine":
            # Cosine annealing
            new_alpha = self.final_alpha - (self.final_alpha - self.initial_alpha) * (
                        1 + math.cos(math.pi * progress)) / 2

        elif self.strategy == "polynomial":
            # Polynomial annealing
            new_alpha = self.initial_alpha + (self.final_alpha - self.initial_alpha) * (progress ** self.power)

        elif self.strategy == "stepwise":
            # Stepwise annealing
            new_alpha = self.initial_alpha + (self.current_step // self.step_size) * self.increment
            new_alpha = min(new_alpha, self.final_alpha)

        elif self.strategy == "sigmoid":
            # Sigmoid annealing
            new_alpha = self.initial_alpha + (self.final_alpha - self.initial_alpha) / (
                        1 + math.exp(-self.k * (self.current_step - self.max_steps / 2)))

        else:
            raise ValueError(f"Unknown annealing strategy: {self.strategy}")

        # Update alpha
        self.alpha = min(new_alpha, self.final_alpha)

        # Ensure alpha does not exceed final_alpha
        return self.alpha


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    # Define parameters
    initial_alpha = 1.0000001
    final_alpha = 2.0
    max_steps = 10000
    strategies = ["linear", "exponential", "cosine", "polynomial", "stepwise", "sigmoid"]

    # Create figure and axis for the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate and plot alpha values for each strategy
    for strategy in strategies:
        scheduler = AlphaScheduler(
            initial_alpha=initial_alpha,
            final_alpha=final_alpha,
            max_steps=max_steps,
            strategy=strategy,
            power=2,  # For polynomial annealing
            step_size=1000,  # For stepwise annealing
            increment=0.1,  # For stepwise annealing
            k=0.001  # For sigmoid annealing
        )

        # Calculate alpha values across all steps
        alphas = [scheduler.step() for _ in range(max_steps)]

        if strategy == "linear" or strategy == "exponential" or strategy == "cosine":
            label = f"{strategy.capitalize()}"
        elif strategy == "polynomial":
            label = f"{strategy.capitalize()}, power=2"
        elif strategy == "stepwise":
            label = f"{strategy.capitalize()}, step_size=1000, inc=0.1"
        else:
            label = f"{strategy.capitalize()}, k=0.001"
        ax.plot(alphas, label=label)

    # Customize the plot
    ax.set_title("Alpha Scheduling Strategies")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Alpha Value")
    ax.legend(title="Strategy")
    ax.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.5)

    # remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Display the plot
    plt.savefig("alpha_scheduler.png", dpi=300, bbox_inches="tight")
