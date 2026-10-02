import math


class BaseSchedule:
    """
    Abstract base class for all schedule classes.
    A schedule defines how a value changes over time based on progress.
    """

    def __call__(self, progress_remaining: float) -> float:
        """
        Calculates the current value of the schedule
        Parameters:
            progress_remaining: A float between 1.0 (start of training) and 0.0 (end of training).
        Returns:
            The current value according to the schedule.
        """
        raise NotImplementedError()


class LinearSchedule(BaseSchedule):
    """
    Linear interpolation between start and end between `progress_remaining` = 1
    and `progress_remaining` = `end_fraction`
    """

    def __init__(self, max_lr: float, min_lr: float, end_fraction: float, warmup_ratio: float = 0.0):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.end_fraction = end_fraction
        self.warmup_ratio = warmup_ratio

    def __call__(self, progress_remaining: float) -> float:
        if progress_remaining > 1.0 - self.warmup_ratio:
            return (1 - progress_remaining) * self.max_lr / self.warmup_ratio
        elif (1 - progress_remaining) > self.end_fraction:
            return self.min_lr
        else:
            return self.max_lr + (1 - progress_remaining - self.warmup_ratio) * (self.max_lr - self.min_lr) / (
                self.warmup_ratio - self.end_fraction
            )


class ConstantSchedule(BaseSchedule):
    def __init__(self, const_val):
        self.const_val = const_val

    def __call__(self, _) -> float:
        return self.const_val


class ExponentialDecaySchedule(BaseSchedule):
    """
    Exponentially decays a value from an initial value towards a minimum value.
    Formula: current_value = initial_value * (decay_rate ^ (progress_made / decay_applicability_fraction)).
    """

    def __init__(
        self, initial_value: float, decay_rate: float, min_value: float = 0.0, decay_applicability_fraction: float = 1.0
    ):

        self.initial_value = initial_value
        self.decay_rate = decay_rate
        self.min_value = min_value
        self.decay_applicability_fraction = decay_applicability_fraction

        self.log_decay_rate = math.log(self.decay_rate)

    def __call__(self, progress_remaining: float) -> float:

        progress_made = 1.0 - progress_remaining

        if progress_made >= self.decay_applicability_fraction:
            current_value = self.initial_value * self.decay_rate
        else:
            effective_progress = progress_made / self.decay_applicability_fraction
            current_value = self.initial_value * math.exp(self.log_decay_rate * effective_progress)

        return max(self.min_value, current_value)
