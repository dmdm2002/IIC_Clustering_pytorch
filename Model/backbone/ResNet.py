import torch
import torch.nn as nn
import torchsummary


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
        # print(model)
        self.model = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        return self.model(x)


# model = ResNet()
# torchsummary.summary(model, (3, 224, 224), device='cpu')