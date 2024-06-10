import pandas
import multiprocessing
import openl3
import soundfile as sf
import os

import numpy as np
import h5py
from torch.utils.data import DataLoader
from IPython import embed
from tqdm import tqdm, trange
from moviepy.video.io.VideoFileClip import VideoFileClip
import argparse
'''
This script is to generate audio and video features from L3 pretrained network for training data.
For details about the L3 net, please refer to https://openl3.readthedocs.io/en/latest/index.html

To run this script,
python extract_openl3_emb.py --input_path '/path/to/file' --dataset_path '/path/of/TAU-urban-audio-visual-scenes-2021-development/'
--output_path '/path/to/save/' --split '[train/val/test]' --hop_size 0.1 --content_type 'env' --input_repr 'mel256' --embedding_size 512

'''


parser = argparse.ArgumentParser(description='Generating audio and video features from L3 network for training data')
parser.add_argument('--input_path', type=str,
                    help='give the file path of train.csv file you generated from split_data.py')
parser.add_argument('--dataset_path', type=str,
                    help='give the path of TAU-urban-audio-visual-scenes-2021-development data')
parser.add_argument('--output_path', type=str,
                    help='give the folder path of where you want to save audio and video features')
parser.add_argument('--split', type=str,
                    help='give the dataset split, train / val / test')
parser.add_argument('--batch_size', type=int, default=32,
                    help='inference batch size')
parser.add_argument('--num_process', type=int, default=4,
                    help='the number of loading data processes')  
parser.add_argument('--hop_size', type=float, default=0.1,
                    help='required by L3 net, hop size needs to be defined. e.g. 0.1 or 0.5')
parser.add_argument('--content_type', type=str, default='env',
                    help='required by L3 net, e.g. env or music')
parser.add_argument('--input_repr', type=str, default='mel256',
                    help='required by L3 net, e.g. linear, mel128, or mel256')
parser.add_argument('--embedding_size', type=int, default=512,
                    help='required by L3 net')
args, _ = parser.parse_known_args()

#### digitalize the target according to the alphabet order#####
classes_dic = {
    'airport': 0,
    'bus': 1,
    'metro': 2,
    'metro_station': 3,
    'park': 4,
    'public_square': 5,
    'shopping_mall': 6,
    'street_pedestrian': 7,
    'street_traffic': 8,
    'tram': 9
}

split = args.split
batch_size = args.batch_size
num_process = args.batch_size

#### set the data path#######
path_csv = args.input_path

#### load the training data using pandas######
df = pandas.read_csv(path_csv, sep="\t")

input_dir_audio = df['filename_audio'].values
input_dir_audio = list(input_dir_audio)
input_dir_audio.sort()

#embed()
####### load the model to extract the audio and video embeddings#####
model_audio = openl3.models.load_audio_embedding_model(content_type=args.content_type,
    input_repr = args.input_repr, embedding_size=args.embedding_size)

model_video = openl3.models.load_image_embedding_model(content_type=args.content_type,
    input_repr = args.input_repr, embedding_size=args.embedding_size)


save_data_audio = os.path.join(args.output_path, 'audio_features_data')
if not os.path.exists(save_data_audio):
    os.makedirs(save_data_audio)
    print("Directory " , save_data_audio ,  " Created ")
else:
    print("Directory " , save_data_audio ,  " already exists")

save_data_video = os.path.join(args.output_path, 'video_features_data')
if not os.path.exists(save_data_video):
    os.makedirs(save_data_video)
    print("Directory ", save_data_video,  " Created ")
else:
    print("Directory ", save_data_video,  " already exists")

first_sample = os.path.join(args.dataset_path, input_dir_audio[0])
audio, SR = sf.read(first_sample)
_, TS = openl3.get_audio_embedding(audio, SR, hop_size=args.hop_size, model=model_audio)


class InferenceAudioDataset(object):

    def __init__(self, input_dir_audios, dataset_path, classes_dic):
        self.input_dir_audios = input_dir_audios
        self.dataset_path = dataset_path
        self.classes_dic = classes_dic

    def __getitem__(self, index):
        input_dir_audio = self.input_dir_audios[index]
        audio_name = os.path.join(self.dataset_path, input_dir_audio)
        label = audio_name.split('/')[-1].split('-')[0]
        label = self.classes_dic[label]
        audio, sr = sf.read(audio_name)

        return {
            "audio": audio,
            "label": label,
            "fname": input_dir_audio
        }

    def __len__(self):
        return len(self.input_dir_audios)


class InferenceVideoDataset(object):

    def __init__(self, input_dir_audios, dataset_path, classes_dic):
        self.input_dir_audios = input_dir_audios
        self.dataset_path = dataset_path
        self.classes_dic = classes_dic

    def __getitem__(self, index):
        input_dir_audio = self.input_dir_audios[index]
        audio_name = os.path.join(self.dataset_path, input_dir_audio)
        label = audio_name.split('/')[-1].split('-')[0]
        label = self.classes_dic[label]
        video_name = input_dir_audio.replace('audio', 'video')
        video_name = video_name.replace('.wav', '.mp4')
        video_name = os.path.join(self.dataset_path, video_name)
        with VideoFileClip(video_name, audio=False) as clip:
            images = []
            for t, frame in clip.iter_frames(with_times=True):
                images.append(frame)
            index = np.linspace(0, len(images) - 1, len(TS))
            index = index.astype(int)
            images = [images[i] for i in list(index)]
            video = np.array(images)
        return {
            "video": video,
            "label": label,
            "fname": input_dir_audio
        }

    def __len__(self):
        return len(self.input_dir_audios)


def collate(data_list):
    if "audio" in data_list[0]:
        audios = []
    elif "video" in data_list[0]:
        videos = []
    labels = []
    fnames = []
    for data in data_list:
        if "audio" in data:
            audios.append(data["audio"])
        elif "video" in data:
            videos.append(data["video"])
        labels.append(data["label"])
        fnames.append(data["fname"])
    if "audio" in data_list[0]:
        return audios, labels, fnames
    elif "video" in data_list[0]:
        return videos, labels, fnames


################create training features #########################

print('generating features data ...')

dataloader_a = DataLoader(
    InferenceAudioDataset(input_dir_audio, args.dataset_path, classes_dic),
    batch_size=args.batch_size,
    num_workers=args.num_process,
    collate_fn=collate
)

hf_audio = h5py.File(os.path.join(save_data_audio, f'{split}.hdf5'), 'w')
for batch in tqdm(dataloader_a):
    audios, labels, fnames = batch
    audio_embs, _ = openl3.get_audio_embedding(audios, SR, hop_size=args.hop_size, model=model_audio)
    for idx in range(len(fnames)):
        hf_audio.create_dataset(
            str(labels[idx]) + '/' + fnames[idx].replace('.wav',''),
            data=audio_embs[idx])
hf_audio.close()

dataloader_v = DataLoader(
    InferenceVideoDataset(input_dir_audio, args.dataset_path, classes_dic),
    batch_size=2,
    num_workers=args.num_process,
    collate_fn=collate
)

hf_video = h5py.File(os.path.join(save_data_video, f'{split}.hdf5'), 'w')
for batch in tqdm(dataloader_v):
    videos, labels, fnames = batch
    video_embs = openl3.get_image_embedding(videos, model=model_video)
    for idx in range(len(fnames)):
        hf_video.create_dataset(
            str(labels[idx]) + '/' + fnames[idx].replace('.wav','').replace('audio', 'video'),
            data=video_embs[idx])
hf_video.close()

################create training features#########################
 