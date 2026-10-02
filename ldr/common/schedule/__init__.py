from typing import Dict
from .scheduler import ConstantSchedule, LinearSchedule, ExponentialDecaySchedule, BaseSchedule

scheduler_factory: Dict[str, BaseSchedule] = {}
scheduler_factory["ConstantSchedule"] = ConstantSchedule
scheduler_factory["LinearSchedule"] = LinearSchedule
scheduler_factory["ExponentialDecaySchedule"] = ExponentialDecaySchedule

__all__ = ["BaseSchedule", "scheduler_factory"]
