import torch.nn as nn
from Model.backbone.ResNet import ResNet


class ClusterNetHeadHead(nn.Module):
    def __init__(self, output_dim, num_sub_heads):
        super(ClusterNetHeadHead, self).__init__()

        self.num_sub_heads = num_sub_heads

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2048, output_dim),
                nn.Softmax(dim=1),
            ) for _ in range(self.num_sub_heads)
        ])

    def forward(self, x, kmeans_use_features=False):
        result = []
        for i in range(self.num_sub_heads):
            if kmeans_use_features:
                result.append(x)

            else:
                result.append(self.heads[i](x))

        return result


class ClusterNetHead(nn.Module):
    def __init__(self, label_dim, aux_output_dim, num_sub_heads):
        super(ClusterNetHead, self).__init__()
        self.backbone = ResNet()

        self.aux_output_dim = aux_output_dim
        self.label_dim = label_dim
        self.num_sub_heads = num_sub_heads

        self.head_A = ClusterNetHeadHead(self.aux_output_dim, num_sub_heads)
        self.head_B = ClusterNetHeadHead(self.label_dim, num_sub_heads)

    def forward(self, x, head="B", kmeans_use_features=False):
        x = self.backbone(x)

        # backbone output Flatten
        x = x.view(x.size(0), -1)

        if head == "A":
            x = self.head_A(x, kmeans_use_features)
        elif head == "B":
            x = self.head_B(x,  kmeans_use_features)
        else:
            assert (False)

        return x