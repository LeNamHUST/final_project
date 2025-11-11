import torch
import torchvision
import torch.nn as nn

class RetinalResnetModel(nn.Module):
    def __init__(self, num_classes, is_trained=True):
        super().__init__()
        self.net = torchvision.models.resnet.resnet50(pretrained=is_trained)
        classifier_input_size = self.net.fc.in_features
        self.net.fc = nn.Sequential(nn.Linear(classifier_input_size, num_classes), nn.Sigmoid())
    def forward(self, images):
        return self.net(images)