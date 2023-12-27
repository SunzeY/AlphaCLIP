import os
import json
import argparse
from cap_metrics import PTBTokenizer, compute_scores

def eval_refcap(gen, gts):
    gts = PTBTokenizer.tokenize(gts)
    gen = PTBTokenizer.tokenize(gen)
    scores, _ = compute_scores(gts, gen)
    #scores, _ = compute_scores(gts, gts)
    print(len(gts), scores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-file", type=str)
    args = parser.parse_args()
    gen = [json.loads(q)['text'] for q in open(args.result_file, 'r')]
    gts = [json.loads(q)['label'] for q in open(args.result_file, 'r')]
    eval_refcap(gen, gts)
