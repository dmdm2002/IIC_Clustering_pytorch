import torch.nn as nn
from Model.backbone.ResNet import ResNet


class IICModel(nn.Module):
    def __init__(self, n_classes, aux_classes, use_kmeans=False):
        super(IICModel, self).__init__()
        self.use_kmeans = use_kmeans
        assert type(use_kmeans) is bool, "The 'use_kmeans' parameter can only have a Boolen format"
        self.n_classes = n_classes
        self.aux_classes = aux_classes

        self.backbone = ResNet()
        self.head_A = nn.Sequential(
            nn.Linear(2048, self.aux_classes),
            nn.Softmax(dim=1)
        )
        self.head_B = nn.Sequential(
            nn.Linear(2048, self.n_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x, head="B"):
        x = self.backbone(x)

        # backbone output Flatten
        x = x.view(x.size(0), -1)
        output = self.head_A(x)

        # if head == "A":
        #     output = self.head_A(x)
        # else:
        #     output = self.head_B(x)

        return output

