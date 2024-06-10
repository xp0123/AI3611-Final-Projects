import argparse
import os
import h5py
import numpy as np
from tqdm import trange
from sklearn.preprocessing import StandardScaler


parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, required=True)

args = parser.parse_args()
audio_feature = os.path.join(args.input_path, "audio_features_data/train.hdf5")
video_feature = os.path.join(args.input_path, "video_features_data/train.hdf5")

all_files = []

def traverse(name, obj):
    if isinstance(obj, h5py.Dataset):
        all_files.append(name)

scaler_a = StandardScaler()
scaler_v = StandardScaler()
with h5py.File(audio_feature, 'r') as hf_a, \
    h5py.File(video_feature, 'r') as hf_v:
    hf_a.visititems(traverse)
    for idx in trange(len(all_files)):
        aid = all_files[idx]
        audio_feat = hf_a[aid][()]
        scaler_a.partial_fit(audio_feat)
        vid = aid.replace('audio','video')
        video_feat = hf_v[vid][()]
        scaler_v.partial_fit(video_feat)

# np.savez(os.path.join(args.input_path, "audio_features_data/global_mean_std.npz"),
np.savez(os.path.join("feature", "audio_features_data/global_mean_std.npz"),
    global_mean=scaler_a.mean_,
    global_std=scaler_a.scale_
)

# np.savez(os.path.join(args.input_path, "video_features_data/global_mean_std.npz"),
np.savez(os.path.join("feature", "video_features_data/global_mean_std.npz"),
    global_mean=scaler_v.mean_,
    global_std=scaler_v.scale_
)
