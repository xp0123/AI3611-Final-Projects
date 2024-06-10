import argparse
import pandas as pd

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

from utils.util import ptb_tokenize

def print_pred(method, score, scores, key_to_refs, key_to_pred):
    print(f"++++++++++++++ {method} ++++++++++++++", file=writer)
    print(f"{method}: {score:.3f}", file=writer)
    best = max(scores)
    worst = min(scores)
    print(f"------- best-{best:.3f} -------", file=writer)
    img_id = id_map[scores.index(best)]
    print("img_id: {}".format(img_id), file=writer)
    print("reference:", key_to_refs[img_id], file=writer)
    print("prediction:", key_to_pred[img_id], file=writer)
    print(f"------- worst-{worst:.3f} -------", file=writer)
    img_id = id_map[scores.index(worst)]
    print("img_id: {}".format(img_id), file=writer)
    print("reference:", key_to_refs[img_id], file=writer)
    print("prediction:", key_to_pred[img_id], file=writer)
    print("+++++++++++++++++++++++++++++++++++", file=writer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prediction_file", type=str)
    parser.add_argument("-r", "--reference_file", type=str)
    parser.add_argument("-o", "--output_file", type=str)
    args = parser.parse_args()
    prediction_df = pd.read_json(args.prediction_file)
    # [n, var_len]
    key_to_pred = dict(zip(prediction_df["img_id"], prediction_df["prediction"]))
    # [n, 5, var_len]
    captions = open(args.reference_file, "r").read().strip().split("\n")
    key_to_refs = {}
    for i, row in enumerate(captions):
        row = row.split("\t")
        row[0] = row[0][: len(row[0]) - 2]  # filename#0 caption
        if row[0] not in key_to_pred:
            continue
        if row[0] in key_to_refs:
            key_to_refs[row[0]].append(row[1])
        else:
            key_to_refs[row[0]] = [row[1]]

    scorers = [Bleu(n=4), Rouge(), Meteor(), Cider(), Spice()]
    key_to_refs = ptb_tokenize(key_to_refs)
    key_to_pred = ptb_tokenize(key_to_pred)

    output = {"SPIDEr": 0}
    id_map = list(key_to_refs.keys())
    spider_scores = []
    with open(args.output_file, "w") as writer:
        for scorer in scorers:
            score, scores = scorer.compute_score(key_to_refs, key_to_pred)
            method = scorer.method()
            output[method] = score
            if method == "Bleu":
                for n in range(4):
                    print_pred(f"Bleu-{n+1}", score[n], scores[n], key_to_refs, key_to_pred)
            elif method == 'SPICE':
                # print("!!!", method)
                scores = [s['All']['f'] for s in scores]
                print_pred(method, score, scores, key_to_refs, key_to_pred)
            else:
                scores = list(scores)
                print_pred(method, score, scores, key_to_refs, key_to_pred)
            if method in ["CIDEr", "SPICE"]:
                print(method)
                spider_scores.append(scores)
                output["SPIDEr"] += score
        output["SPIDEr"] /= 2
        print(len(spider_scores[0]), len(spider_scores[1]))
        len_ = len(spider_scores[0])
        spider_scores_ = [(spider_scores[0][i] + spider_scores[1][i]) / 2 for i in range(len_)]
        print_pred("SPIDEr", output["SPIDEr"], spider_scores_, key_to_refs, key_to_pred)
        print(f"SPIDEr: {output['SPIDEr']:.3f}", file=writer)

