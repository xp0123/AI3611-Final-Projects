import torch
import torch.nn as nn


def linear_softmax_pooling(x):
    return (x ** 2).sum(1) / x.sum(1)


class Crnn(nn.Module):
    def __init__(self, num_freq, num_class):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     num_freq: int, mel frequency bins
        #     num_class: int, the number of output classes
        ##############################
        super(Crnn, self).__init__()
        self.num_freq = num_freq
        self.num_class = num_class
        self.batch_norm = nn.BatchNorm1d(num_freq)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.biGRU = nn.GRU(128 * 2, 256, bidirectional = True, batch_first = True)
        self.fc = nn.Linear(256 * 2, num_class)

    def detection(self, x):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     x: [batch_size, time_steps, num_freq]
        # Return:
        #     frame_prob: [batch_size, time_steps, num_class]
        ##############################
        batch_size, time_steps, num_freq = x.shape
        # print("x.shape", x.shape)
        x_batch_norm = self.batch_norm(x.permute(0, 2, 1)).view(batch_size, 1, num_freq, time_steps)
        # print("x_batch_norm.shape", x_batch_norm.shape)
        f1 = self.conv_block1(x_batch_norm)
        f2 = self.conv_block2(f1)
        f3 = self.conv_block3(f2)
        f4 = self.conv_block4(f3) # 64*128*4*31
        f5 = self.conv_block5(f4) # 64*128*2*15
        d = f5.shape[-1]
        f, _ = self.biGRU(f5.view(batch_size, -1, d).permute(0, 2, 1))
        f = self.fc(f)
        f = torch.sigmoid(f)
        out = nn.functional.interpolate(f.permute(0,2,1), time_steps).permute(0,2,1)
        return out

    def forward(self, x):
        frame_prob = self.detection(x)  # (batch_size, time_steps, class_num)
        clip_prob = linear_softmax_pooling(frame_prob)  # (batch_size, class_num)
        '''(samples_num, feature_maps)'''
        return {
            'clip_prob': clip_prob, 
            'frame_prob': frame_prob
        }
