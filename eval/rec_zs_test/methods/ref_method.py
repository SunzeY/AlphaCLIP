"""Base class for a method for doing referring expressions."""

from typing import Dict, Any
from abc import ABCMeta, abstractmethod


class RefMethod(metaclass=ABCMeta):
    @abstractmethod
    def execute(self, caption: str, env: "Environment") -> Dict[str, Any]:
        return NotImplemented

    def get_stats(self) -> Dict[str, Any]:
        return {}
