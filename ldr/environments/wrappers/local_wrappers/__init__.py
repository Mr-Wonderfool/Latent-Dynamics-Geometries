from .clean_observation_wrapper import CleanObservationWrapper
from .limit_angle_torque_wrapper import LimitAngleTorqueWrapper
from .physics_wrapper import (
    HopperDynamicsWrapper,
    Walker2dDynamicsWrapper,
    HalfCheetahDynamicsWrapper,
    AntDynamicsWrapper,
)
from .extract_env_fractor_wrapper import ExtractEnvFactorWrapper
from .bodyframe_wrapper import BodyFrameWrapper

local_wrapper_factory = {}
local_wrapper_factory["HopperDynamicsWrapper"] = HopperDynamicsWrapper
local_wrapper_factory["Walker2dDynamicsWrapper"] = Walker2dDynamicsWrapper
local_wrapper_factory["HalfCheetahDynamicsWrapper"] = HalfCheetahDynamicsWrapper
local_wrapper_factory["AntDynamicsWrapper"] = AntDynamicsWrapper

local_wrapper_factory["CleanObservationWrapper"] = CleanObservationWrapper
local_wrapper_factory["LimitAngleTorqueWrapper"] = LimitAngleTorqueWrapper

local_wrapper_factory["ExtractEnvFactorWrapper"] = ExtractEnvFactorWrapper

local_wrapper_factory["BodyFrameWrapper"] = BodyFrameWrapper

__all__ = ["local_wrapper_factory"]
