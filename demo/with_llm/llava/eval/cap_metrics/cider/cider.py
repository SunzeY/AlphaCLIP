# Filename: cider.py
#
# Description: Describes the class to compute the CIDEr (Consensus-Based Image Description Evaluation) Metric
#               by Vedantam, Zitnick, and Parikh (http://arxiv.org/abs/1411.5726)
#
# Creation Date: Sun Feb  8 14:16:54 2015
#
# Authors: Ramakrishna Vedantam <vrama91@vt.edu> and Tsung-Yi Lin <tl483@cornell.edu>

from .cider_scorer import CiderScorer


class Cider:
    """Main Class to compute the CIDEr metric."""

    def __init__(self, gts=None, n=4, sigma=6.0):
        # set cider to sum over 1 to 4-grams
        self._n = n
        # set the standard deviation parameter for gaussian penalty
        self._sigma = sigma
        self.doc_frequency = None
        self.ref_len = None
        if gts is not None:
            tmp_cider = CiderScorer(gts, n=self._n, sigma=self._sigma)
            self.doc_frequency = tmp_cider.doc_frequency
            self.ref_len = tmp_cider.ref_len

    def compute_score(self, gts, res):
        """Main function to compute CIDEr score.

        :param  gts (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
                res (dict)  : dictionary with key <image> and value <tokenized reference sentence>
        :return: cider (float) : computed CIDEr score for the corpus
        """
        assert (gts.keys() == res.keys())
        cider_scorer = CiderScorer(
            gts,
            test=res,
            n=self._n,
            sigma=self._sigma,
            doc_frequency=self.doc_frequency,
            ref_len=self.ref_len)
        return cider_scorer.compute_score()

    def __str__(self):
        return 'CIDEr'
