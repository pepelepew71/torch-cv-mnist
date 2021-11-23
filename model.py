"""
"""

import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class SimpleNet4(nn.Module):

    def __init__(self):
        super(SimpleNet4, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),  # 64 x24 x 24
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),  # 64 x 12 x 12
            nn.ReLU(True),
            nn.Conv2d(64, 50, kernel_size=5),  # 50 x 8 x 8
            nn.BatchNorm2d(50),
            nn.Dropout2d(),
            nn.MaxPool2d(2),  # 50 x 4 x 4
            nn.ReLU(True),
            nn.Flatten(),   # 800
            nn.Linear(800, 512),  # 512
            nn.BatchNorm1d(512),
            nn.Dropout(),
            nn.ReLU(True)
        )
        self.label_classifier = nn.Sequential(
            nn.Linear(512, 100),  # 100
            nn.BatchNorm1d(100),
            nn.Dropout(),
            nn.ReLU(True),
            nn.Linear(100, 10),  # 10
            nn.Softmax(dim=1)
        )
        self.apply(weights_init)

    def forward(self, input_data):
        feature = self.feature(input_data)
        out_label = self.label_classifier(feature)
        return out_label


if __name__ == "__main__":

    simple_net_4 = SimpleNet4()
    print(simple_net_4)
    from torchsummary import summary
    print(summary(simple_net_4, (3,28,28), device="cpu"))
