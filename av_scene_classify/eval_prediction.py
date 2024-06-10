import argparse
from pathlib import Path

from sklearn.metrics import accuracy_score, log_loss, classification_report
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("prediction", type=str)
parser.add_argument("label", type=str, help="path to fold1_evaluate.csv")


keys = ['airport',
        'bus',
        'metro',
        'metro_station',
        'park',
        'public_square',
        'shopping_mall',
        'street_pedestrian',
        'street_traffic',
        'tram']

scene_to_idx = { scene: idx for idx, scene in enumerate(keys) }

args = parser.parse_args()
label_df = pd.read_csv(args.label, sep="\t")

label_df["aid"] = label_df["filename_audio"].apply(lambda x: Path(x).stem)

aid_to_label = dict(zip(label_df["aid"], label_df["scene_label"]))

targets = []
probs = []
preds = []

pred_df = pd.read_csv(args.prediction, sep="\t")
for idx, row in pred_df.iterrows():
    aid = row["aid"]
    pred = row["scene_pred"]
    targets.append(scene_to_idx[aid_to_label[aid]])
    preds.append(scene_to_idx[pred])

targets = np.array(targets)
preds = np.array(preds)

for key in keys:
    probs.append(pred_df[key].values)

probs = np.stack(probs, axis=1)
print(classification_report(targets, preds, target_names=keys))

acc = accuracy_score(targets, preds)
print('  ')
print(f'accuracy: {acc:.3f}')
logloss = log_loss(targets, probs)
print(f'overall log loss: {logloss:.3f}')
