from .bleu import Bleu
from .cider import Cider
from .meteor import Meteor
from .rouge import Rouge
from .tokenizer import PTBTokenizer






def compute_scores(gts, gen):
    metrics = (Bleu(), Meteor(), Rouge(), Cider())
    #metrics = (Bleu(), Rouge(), Cider())
    all_score = {}
    all_scores = {}
    for metric in metrics:
        score, scores = metric.compute_score(gts, gen)
        all_score[str(metric)] = score
        all_scores[str(metric)] = scores

    return all_score, all_scores
