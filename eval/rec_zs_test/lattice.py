"""Implement lattice interface."""

from overrides import overrides
import numpy as np
from abc import ABCMeta, abstractmethod


class Lattice(metaclass=ABCMeta):

    """Abstract base class representing a complemented lattice."""

    @classmethod
    @abstractmethod
    def join(cls, probs1: np.ndarray, probs2: np.ndarray) -> np.ndarray:
        return NotImplemented
    
    @classmethod
    @abstractmethod
    def meet(cls, probs1: np.ndarray, probs2: np.ndarray) -> np.ndarray:
        return NotImplemented
    
    @classmethod
    @abstractmethod
    def join_reduce(cls, probs: np.ndarray) -> np.ndarray:
        return NotImplemented

    @classmethod
    @abstractmethod
    def meet_reduce(cls, probs: np.ndarray) -> np.ndarray:
        return NotImplemented


class Product(Lattice):
    """Lattice where meet=prod and sum is defined accordingly.
    
    Equivalent to assuming independence, more or less.
    """

    eps = 1e-9

    @classmethod
    @overrides
    def join(cls, probs1: np.ndarray, probs2: np.ndarray) -> np.ndarray:
        return probs1 + probs2 - cls.meet(probs1, probs2)

    @classmethod
    @overrides
    def meet(cls, probs1: np.ndarray, probs2: np.ndarray) -> np.ndarray:
        return probs1 * probs2

    @classmethod
    @overrides
    def join_reduce(cls, probs: np.ndarray) -> np.ndarray:
        """Assumes disjoint events."""
        # return cls.comp(cls.meet_reduce(cls.comp(probs)))
        return np.sum(probs, axis=-1)

    @classmethod
    @overrides
    def meet_reduce(cls, probs: np.ndarray) -> np.ndarray:
        return np.prod(probs, axis=-1)

    @classmethod
    def comp(cls, probs):
        return 1 - probs

    @classmethod
    def normalize(cls, probs):
        """Normalize a distribution by dividing by the total mass."""
        return probs / np.sum(probs + cls.eps, axis=-1)
