from typing import Dict
from dataclasses import dataclass, field

from ldr.common.base_config import BaseConfig


@dataclass
class DomainAdapationConfig(BaseConfig):
    adapt_wrappers: Dict[str, Dict] = field(default_factory=dict)
