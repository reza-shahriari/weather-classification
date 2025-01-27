import torch.nn as nn
import torchvision.models as models

class WeatherClassifier(nn.Module):
    def __init__(self, num_classes):
        super(WeatherClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)
