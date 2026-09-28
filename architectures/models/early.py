import torch.nn as nn


class EarlyFusionModel(nn.Module):

    def __init__(self, encoder_class, num_classes, num_channels, hyperparameters):
        super().__init__()
        num_filters = hyperparameters["num_filters"]
        self.encoder = encoder_class(num_channels, num_filters)
        self.classifier = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        # x: (B, C, L)
        features = self.encoder(x)
        # features: (B, F)
        logits = self.classifier(features)
        # logits: (B, K)
        return {"logits": logits, "features": features}
