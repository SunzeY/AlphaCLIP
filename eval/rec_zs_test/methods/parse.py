"""Use spatial relations extracted from the parses."""

from typing import Dict, Any, Callable, List, Tuple, NamedTuple
from numbers import Number
from collections import defaultdict
from overrides import overrides
import numpy as np
import spacy
from spacy.tokens.token import Token
from spacy.tokens.span import Span
from argparse import Namespace

from .ref_method import RefMethod
from lattice import Product as L
from heuristics import Heuristics
from entity_extraction import Entity, expand_chunks


def get_conjunct(ent, chunks, heuristics: Heuristics) -> Entity:
    """If an entity represents a conjunction of two entities, pull them apart."""
    head = ent.head.root  # Not ...root.head. Confusing names here.
    if not any(child.text == "and" for child in head.children):
        return None
    for child in head.children:
        if child.i in chunks and head.i is not child.i:
            return Entity.extract(child, chunks, heuristics)
    return None


class Parse(RefMethod):
    """An REF method that extracts and composes predicates, relations, and superlatives from a dependency parse.

    The process is as follows:
        1. Use spacy to parse the document.
        2. Extract a semantic entity tree from the parse.
        3. Execute the entity tree to yield a distribution over boxes."""

    nlp = spacy.load('en_core_web_sm')

    def __init__(self, args: Namespace = None):
        self.args = args
        self.box_area_threshold = args.box_area_threshold
        self.baseline_threshold = args.baseline_threshold
        self.temperature = args.temperature
        self.superlative_head_only = args.superlative_head_only
        self.expand_chunks = args.expand_chunks
        self.branch = not args.parse_no_branch
        self.possessive_expand = not args.possessive_no_expand

        # Lists of keyword heuristics to use.
        self.heuristics = Heuristics(args)

        # Metrics for debugging relation extraction behavor.
        self.counts = defaultdict(int)

    @overrides
    def execute(self, caption: str, env: "Environment") -> Dict[str, Any]:
        """Construct an `Entity` tree from the parse and execute it to yield a distribution over boxes."""
        # Start by using the full caption, as in Baseline.
        probs = env.filter(caption, area_threshold=self.box_area_threshold, softmax=True) 
        ori_probs = probs

        # Extend the baseline using parse stuff.
        doc = self.nlp(caption) 
        head = self.get_head(doc) 
        chunks = self.get_chunks(doc) 
        if self.expand_chunks:
            chunks = expand_chunks(doc, chunks)
        entity = Entity.extract(head, chunks, self.heuristics) 

        # If no head noun is found, take the first one.
        if entity is None and len(list(doc.noun_chunks)) > 0:
            head = list(doc.noun_chunks)[0]
            entity = Entity.extract(head.root.head, chunks, self.heuristics)
            self.counts["n_0th_noun"] += 1

        # If we have found some head noun, filter based on it.
        if entity is not None and (any(any(token.text in h.keywords for h in self.heuristics.relations+self.heuristics.superlatives) for token in doc) or not self.branch):
            ent_probs, texts = self.execute_entity(entity, env, chunks)
            probs = L.meet(probs, ent_probs)
        else:
            texts = [caption]
            self.counts["n_full_expr"] += 1

        if len(ori_probs) == 1:
            probs = ori_probs

        self.counts["n_total"] += 1
        pred = np.argmax(probs)
        return {
            "probs": probs,
            "pred": pred,
            "box": env.boxes[pred],
            "texts": texts
        }

    def execute_entity(self,
                       ent: Entity,
                       env: "Environment",
                       chunks: Dict[int, Span],
                       root: bool = True,
                      ) -> np.ndarray:
        """Execute an `Entity` tree recursively, yielding a distribution over boxes."""
        self.counts["n_rec"] += 1
        probs = [1, 1]
        head_probs = probs

        # Only use relations if the head baseline isn't certain.
        if len(probs) == 1 or len(env.boxes) == 1:
            return probs, [ent.text]

        m1, m2 = probs[:2] # probs[(-probs).argsort()[:2]]
        text = ent.text
        rel_probs = []
        if self.baseline_threshold == float("inf") or m1 < self.baseline_threshold * m2:
            self.counts["n_rec_rel"] += 1
            for tokens, ent2 in ent.relations:
                self.counts["n_rel"] += 1
                rel = None
                # Heuristically decide which spatial relation is represented.
                for heuristic in self.heuristics.relations:
                    if any(tok.text in heuristic.keywords for tok in tokens):
                        rel = heuristic.callback(env)
                        self.counts[f"n_rel_{heuristic.keywords[0]}"] += 1
                        break
                # Filter and normalize by the spatial relation.
                if rel is not None:
                    probs2 = self.execute_entity(ent2, env, chunks, root=False)
                    events = L.meet(np.expand_dims(probs2, axis=0), rel)
                    new_probs = L.join_reduce(events)
                    rel_probs.append((ent2.text, new_probs, probs2))
                    continue

                # This case specifically handles "between", which takes two noun arguments.
                rel = None
                for heuristic in self.heuristics.ternary_relations:
                    if any(tok.text in heuristic.keywords for tok in tokens):
                        rel = heuristic.callback(env)
                        self.counts[f"n_rel_{heuristic.keywords[0]}"] += 1
                        break
                if rel is not None:
                    ent3 = get_conjunct(ent2, chunks, self.heuristics)
                    if ent3 is not None:
                        probs2 = self.execute_entity(ent2, env, chunks, root=False)
                        probs2 = np.expand_dims(probs2, axis=[0, 2])
                        probs3 = self.execute_entity(ent3, env, chunks, root=False)
                        probs3 = np.expand_dims(probs3, axis=[0, 1])
                        events = L.meet(L.meet(probs2, probs3), rel)
                        new_probs = L.join_reduce(L.join_reduce(events))
                        probs = L.meet(probs, new_probs)
                    continue
                # Otherwise, treat the relation as a possessive relation.
                if not self.args.no_possessive:
                    if self.possessive_expand:
                        text = ent.expand(ent2.head)
                    else:
                        text += f' {" ".join(tok.text for tok in tokens)} {ent2.text}'
                    #poss_probs = self._filter(text, env, root=root, expand=.3)
            probs = self._filter(text, env, root=root)
            texts = [text]
            return_probs = [(probs.tolist(), probs.tolist())]
            for (ent2_text, new_probs, ent2_only_probs) in rel_probs:
                probs = L.meet(probs, new_probs)
                probs /= probs.sum()
                texts.append(ent2_text)
                return_probs.append((probs.tolist(), ent2_only_probs.tolist()))

        # Only use superlatives if thresholds work out.
        m1, m2 = probs[(-probs).argsort()[:2]]
        if m1 < self.baseline_threshold * m2:
            self.counts["n_rec_sup"] += 1
            for tokens in ent.superlatives:
                self.counts["n_sup"] += 1
                sup = None
                for heuristic_index, heuristic in enumerate(self.heuristics.superlatives):
                    if any(tok.text in heuristic.keywords for tok in tokens):
                        texts.append('sup:'+' '.join([tok.text for tok in tokens if tok.text in heuristic.keywords]))
                        sup = heuristic.callback(env)
                        self.counts[f"n_sup_{heuristic.keywords[0]}"] += 1
                        break
                if sup is not None:
                    # Could use `probs` or `head_probs` here?
                    precond = head_probs if self.superlative_head_only else probs
                    probs = L.meet(np.expand_dims(precond, axis=1)*np.expand_dims(precond, axis=0), sup).sum(axis=1)
                    probs = probs / probs.sum()
                    return_probs.append((probs.tolist(), None))

        if root:
            assert len(texts) == len(return_probs)
            return probs, (texts, return_probs, tuple(str(chunk) for chunk in chunks.values()))
        return probs

    def get_head(self, doc) -> Token:
        """Return the token that is the head of the dependency parse. """
        for token in doc:
            if token.head.i == token.i:
                return token
        return None

    def get_chunks(self, doc) -> Dict[int, Any]:
        """Return a dictionary mapping sentence indices to their noun chunk."""
        chunks = {}
        for chunk in doc.noun_chunks:
            for idx in range(chunk.start, chunk.end):
                chunks[idx] = chunk
        return chunks

    @overrides
    def get_stats(self) -> Dict[str, Number]:
        """Summary statistics that have been tracked on this object."""
        stats = dict(self.counts)
        n_rel_caught = sum(v for k, v in stats.items() if k.startswith("n_rel_"))
        n_sup_caught = sum(v for k, v in stats.items() if k.startswith("n_sup_"))
        stats.update({
            "p_rel_caught": n_rel_caught / (self.counts["n_rel"] + 1e-9),
            "p_sup_caught": n_sup_caught / (self.counts["n_sup"] + 1e-9),
            "p_rec_rel": self.counts["n_rec_rel"] / (self.counts["n_rec"] + 1e-9),
            "p_rec_sup": self.counts["n_rec_sup"] / (self.counts["n_rec"] + 1e-9),
            "p_0th_noun": self.counts["n_0th_noun"] / (self.counts["n_total"] + 1e-9),
            "p_full_expr": self.counts["n_full_expr"] / (self.counts["n_total"] + 1e-9),
            "avg_rec": self.counts["n_rec"] / self.counts["n_total"],
        })
        return stats

    def _filter(self,
                caption: str,
                env: "Environment",
                root: bool = False,
                expand: float = None,
               ) -> np.ndarray:
        """Wrap a filter call in a consistent way for all recursions."""
        kwargs = {
            "softmax": not self.args.sigmoid,
            "temperature": self.args.temperature,
        }
        if root:
            return env.filter(caption, area_threshold=self.box_area_threshold, **kwargs)
        else:
            return env.filter(caption, **kwargs)
