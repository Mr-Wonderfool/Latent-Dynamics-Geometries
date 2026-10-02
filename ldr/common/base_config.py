import os
from torch import optim
from typing import Optional, Type
from dataclasses import is_dataclass, fields

from .utils.params_manager import ParamsManager
from .schedule import BaseSchedule, scheduler_factory


class BaseConfig:
    @classmethod
    def fromYaml(cls, config_file: str, config_section: Optional[str] = None):
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"No yaml file found at {config_file}")

        parsed_dict = ParamsManager.parse(config_file)
        if config_section:
            if config_section not in parsed_dict:
                raise KeyError(f"Section '{config_section}' not found in config file.")
            config_dict = parsed_dict[config_section]
        else:
            config_dict = parsed_dict

        return cls.fromDict(config_dict)

    @classmethod
    def fromDict(cls, config_dict: dict) -> "BaseConfig":
        # recursively instantiate nested dataclasses
        nested_instances = {}
        for f in fields(cls):
            name = f.name
            field_type = f.type
            value = config_dict.get(name)

            if is_dataclass(field_type) and isinstance(value, dict):
                nested_instances[name] = field_type.fromDict(value)
            elif value is not None:
                if field_type in (int, float, bool):
                    nested_instances[name] = field_type(value)
                else:
                    nested_instances[name] = value

        return cls(**nested_instances)


class BaseOptimizerConfig(BaseConfig):
    @classmethod
    def optimizerFromName(cls, optimizer_name: str) -> Type[optim.Optimizer]:
        if optimizer_name == "Adam":
            return optim.Adam
        elif optimizer_name == "SGD":
            return optim.SGD
        elif optimizer_name == "AdamW":
            return optim.AdamW
        else:
            raise NotImplementedError

    @classmethod
    def schedulerFromName(cls, scheduler_name: str, scheduler_kwargs: dict) -> BaseSchedule:
        if scheduler_name in scheduler_factory:
            return scheduler_factory[scheduler_name](**scheduler_kwargs)
        else:
            raise ValueError(f"Unsupported schedule type, choices are: {scheduler_factory.keys()}")
