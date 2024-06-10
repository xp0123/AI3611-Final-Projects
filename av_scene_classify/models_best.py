import torch
import torch.nn as nn


class MeanConcatDense(nn.Module):

    def __init__(self, audio_emb_dim, video_emb_dim, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.audio_embed = nn.Sequential(
            nn.Linear(audio_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, self.num_classes)
        )
        self.video_embed = nn.Sequential(
            nn.Linear(video_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, self.num_classes)
        )
        self.embed = nn.Sequential(
            nn.Linear(audio_emb_dim + video_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256)
        )
        self.outputlayer = nn.Sequential(
            nn.Linear(256, 128),
            nn.Linear(128, self.num_classes),
        )
        self.audio_output = nn.Sequential(
            nn.Linear(128, 128),
            nn.Linear(128, self.num_classes),
        )
        self.video_output = nn.Sequential(
            nn.Linear(128, 128),
            nn.Linear(128, self.num_classes),
        )
        self.weight = nn.Parameter(
            torch.zeros((1, self.num_classes)),
            requires_grad=True,
        )
        with torch.no_grad():
            self.weight.uniform_(-0.1, 0.1)
    
    def forward(self, audio_feat, video_feat, method='concat'):
        # audio_feat: [batch_size, time_steps, feat_dim]
        # video_feat: [batch_size, time_steps, feat_dim]

        if method == 'early':
            audio_emb = audio_feat.mean(1)
            video_emb = video_feat.mean(1)
            embed = torch.cat((audio_emb, video_emb), 1)
            embed = self.embed(embed)
            output = self.outputlayer(embed)
            return output

        audio_emb = audio_feat.mean(1)
        audio_emb = self.audio_embed(audio_emb)

        video_emb = video_feat.mean(1)
        video_emb = self.video_embed(video_emb)
        
        if method == 'concat':
            # embed = torch.cat((audio_emb, video_emb), 1)
            # output = self.outputlayer(embed)
            output = audio_emb + video_emb * self.weight
        elif method == 'late':
            audio_out = self.audio_output(audio_emb)
            video_out = self.video_output(video_emb)
            output = audio_out + video_out * self.weight
        return output

