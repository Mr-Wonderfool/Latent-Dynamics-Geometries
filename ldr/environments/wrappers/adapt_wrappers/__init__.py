from .cripple_joint_wrapper import CrippleJointWrapper
from .lag_reward_wrapper import LagRewardWrapper

adapt_wrapper_factory = {}
adapt_wrapper_factory["CrippleJointWrapper"] = CrippleJointWrapper
adapt_wrapper_factory["LagRewardWrapper"] = LagRewardWrapper


__all__ = ["adapt_wrapper_factory"]
