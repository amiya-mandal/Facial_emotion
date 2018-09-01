import torch.nn as nn
import torch.nn.functional as F

class EmoModel(nn.Module):

    def __init__(self):

        super(EmoModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64,kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4),
            nn.ReLU(True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3),
            nn.ReLU(True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2),
            nn.ReLU(True)
        )

        self.Maxpool = nn.MaxPool2d(kernel_size=3)

        self.fc6 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(25088, 4048),
            nn.ReLU(True),
        )

        self.fc7 = nn.Sequential(
            nn.Linear(4048, 2024),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )
        self.fc8 = nn.Sequential(
            nn.Linear(2024, 6),
        )

    def forward(self, input):
        # print(input.shape)
        out = self.conv1(input)
        # print(out.shape)
        out = self.conv2(out)
        # print(out.shape)
        out = self.conv3(out)
        # print(out.shape)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.Maxpool(out)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.fc6(out)
        # print(out.shape)
        out = self.fc7(out)
        # print(out.shape)
        out = F.log_softmax(self.fc8(out))
        # print(out.shape)
        return out