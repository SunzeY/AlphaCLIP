"""Heuristic rules used to extract and execute entity parses."""

from typing import Callable, List, NamedTuple
from argparse import Namespace
import numpy as np


class RelHeuristic(NamedTuple):
    keywords: List[str]
    callback: Callable[["Environment"], np.ndarray]


class Heuristics:
    """A class defining heuristics that can be enabled/disabled."""

    RELATIONS = [
        RelHeuristic(["left", "west"], lambda env: env.left_of()),
        RelHeuristic(["right", "east"], lambda env: env.right_of()),
        RelHeuristic(["above", "north", "top", "back", "behind"], lambda env: env.above()),
        RelHeuristic(["below", "south", "under", "front"], lambda env: env.below()),
        RelHeuristic(["bigger", "larger", "closer"], lambda env: env.bigger_than()),
        RelHeuristic(["smaller", "tinier", "further"], lambda env: env.smaller_than()),
        RelHeuristic(["inside", "within", "contained"], lambda env: env.within()),
    ]

    TERNARY_RELATIONS = [
        RelHeuristic(["between"], lambda env: env.between()),
    ]

    SUPERLATIVES = [
        RelHeuristic(["left", "west", "leftmost", "western"], lambda env: env.left_of()),
        RelHeuristic(["right", "rightmost", "east", "eastern"], lambda env: env.right_of()),
        RelHeuristic(["above", "north", "top"], lambda env: env.above()),
        RelHeuristic(["below", "south", "underneath", "front"], lambda env: env.below()),
        RelHeuristic(["bigger", "biggest", "larger", "largest", "closer", "closest"], lambda env: env.bigger_than()),
        RelHeuristic(["smaller", "smallest", "tinier", "tiniest", "further", "furthest"], lambda env: env.smaller_than()),
    ]
    OPPOSITES = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}

    NULL_KEYWORDS = ["part", "image", "side", "picture", "half", "region", "section"]

    EMPTY = []

    def __init__(self, args: Namespace = None):
        self.enable_relations = not args or not args.no_rel
        self.enable_superlatives = not args or not args.no_sup
        self.enable_nulls = not args or not args.no_null
        self.enable_ternary = not args or args.ternary

    @property
    def relations(self) -> List[RelHeuristic]:
        return self.RELATIONS if self.enable_relations else self.EMPTY
    
    @property
    def ternary_relations(self) -> List[RelHeuristic]:
        return self.TERNARY_RELATIONS if self.enable_ternary else self.EMPTY

    @property
    def superlatives(self) -> List[RelHeuristic]:
        return self.SUPERLATIVES if self.enable_superlatives else self.EMPTY

    @property
    def opposites(self):
        return self.OPPOSITES
    
    @property
    def null_keywords(self) -> List[str]:
        return self.NULL_KEYWORDS if self.enable_nulls else self.EMPTY
