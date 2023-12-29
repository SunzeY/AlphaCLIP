"""A naive baseline method: just pass the full expression to CLIP."""

from overrides import overrides
from typing import Dict, Any, List
import numpy as np
import torch
import spacy
from argparse import Namespace

from .ref_method import RefMethod
from lattice import Product as L


class Baseline(RefMethod):
    """CLIP-only baseline where each box is evaluated with the full expression."""

    nlp = spacy.load('en_core_web_sm')

    def __init__(self, args: Namespace):
        self.args = args
        self.box_area_threshold = args.box_area_threshold
        self.batch_size = args.batch_size
        self.batch = []

    @overrides
    def execute(self, caption: str, env: "Environment") -> Dict[str, Any]:
        chunk_texts = self.get_chunk_texts(caption)
        probs = env.filter(caption, area_threshold = self.box_area_threshold, softmax=True)
        if self.args.baseline_head:
            probs2 = env.filter(chunk_texts[0], area_threshold = self.box_area_threshold, softmax=True)
            probs = L.meet(probs, probs2)
        pred = np.argmax(probs)
        return {
            "probs": probs,
            "pred": pred,
            "box": env.boxes[pred],
        }

    def get_chunk_texts(self, expression: str) -> List:
        doc = self.nlp(expression)
        head = None
        for token in doc:
            if token.head.i == token.i:
                head = token
                break
        head_chunk = None
        chunk_texts = []
        for chunk in doc.noun_chunks:
            if head.i >= chunk.start and head.i < chunk.end:
                head_chunk = chunk.text
            chunk_texts.append(chunk.text)
        if head_chunk is None:
            if len(list(doc.noun_chunks)) > 0:
                head_chunk = list(doc.noun_chunks)[0].text
            else:
                head_chunk = expression
        return [head_chunk] + [txt for txt in chunk_texts if txt != head_chunk]
