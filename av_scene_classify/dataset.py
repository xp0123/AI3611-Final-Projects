import numpy as np
import torch
from torch.utils import data
import h5py    

class SceneDataset(data.Dataset):

    def __init__(self,
                 audio_feature,
                 video_feature,
                 audio_transform=None,
                 video_transform=None):
        super().__init__()
        self.audio_feature = audio_feature
        self.video_feature = video_feature
        self.audio_transform = audio_transform
        self.video_transform = video_transform
        self.audio_hf = None
        self.video_hf = None

        self.all_files = []

        def traverse(name, obj):
            if isinstance(obj, h5py.Dataset):
                self.all_files.append(name)

        hf = h5py.File(self.audio_feature, 'r')
        hf.visititems(traverse)
        hf.close()
        print("Finish loading indexes")

    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, index):
        if self.audio_hf is None:
            self.audio_hf = h5py.File(self.audio_feature, 'r')
        if self.video_hf is None:
            self.video_hf = h5py.File(self.video_feature, 'r')

        audio_feat = []
        aid = self.all_files[index]
        audio_feat = self.audio_hf[aid][:96, :]
        # import pdb; pdb.set_trace()
        if self.audio_transform:
            audio_feat = self.audio_transform(audio_feat)
        
        vid = aid.replace("audio", "video")
        video_feat = self.video_hf[vid][:96, :]
        if self.video_transform:
            video_feat = self.video_transform(video_feat)
        
        target = int(aid.split('/')[0])

        audio_feat = torch.as_tensor(audio_feat).float()
        video_feat = torch.as_tensor(video_feat).float()
        target = torch.as_tensor(target).long()
        # print(audio_feat.shape, video_feat.shape)
        return {
            "aid": aid.split("/")[-1],
            "audio_feat": audio_feat,
            "video_feat": video_feat,
            "target": target
        }
    
