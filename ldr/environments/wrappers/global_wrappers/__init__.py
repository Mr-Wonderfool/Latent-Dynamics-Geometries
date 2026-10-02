from .global_dynamics_id_wrapper import GlobalDynamicsIDWrapper

global_wrapper_factory = {}
global_wrapper_factory["GlobalDynamicsIDWrapper"] = GlobalDynamicsIDWrapper

__all__ = ["global_wrapper_factory"]
