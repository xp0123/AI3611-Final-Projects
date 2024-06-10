from utils.metrics import accuracy_fn, bleu_score_fn
import pandas as pd

if __name__ == "__main__":
    prediction_df = pd.read_json('experiments/resnet101_attention/resnet101_attention_b128_emd300_predictions.json')
    #[n, var_len]
    predictions = prediction_df['prediction'].to_list()
    predictions = [pred.split() for pred in predictions]
    #[n, 5, var_len]
    references = prediction_df["references"].to_list()
    references = [[ref.split() for ref in reference] for reference in references]

    corpus_bleu_score_fn = bleu_score_fn(4, 'corpus')
    bleu = [0.0] * 5
    for i in (1, 2, 3, 4):
        bleu[i] = corpus_bleu_score_fn(reference_corpus=references, candidate_corpus=predictions, n=i)
        print("B{}: {:.3f}".format(i, bleu[i]))
