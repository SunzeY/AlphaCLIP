"""A naive baseline method: just pass the full expression to CLIP."""

from overrides import overrides
from typing import Dict, Any
import random
from argparse import Namespace

import numpy as np

from .ref_method import RefMethod


class Random(RefMethod):
    """CLIP-only baseline where each box is evaluated with the full expression."""

    def __init__(self, args: Namespace):
        self.box_area_threshold = args.box_area_threshold

    @overrides
    def execute(self, caption: str, env: "Environment") -> Dict[str, Any]:
        probs = env.filter_area(self.box_area_threshold)*env.uniform()
        random_ordering = list(range(len(env.boxes)))
        random.shuffle(random_ordering)
        random_ordering = np.array(random_ordering)
        pred = np.argmax(probs*random_ordering)
        return {
            "probs": probs.tolist(),
            "pred": int(pred),
            "text": caption.lower()
        }
