import torch.nn as nn
from torch.autograd import Function


class DiversifyModel(nn.Module):

    def __init__(self, architecture_class, num_classes, num_channels, hyperparameters):
        super().__init__()
        self.num_domains = hyperparameters["K"]
        self.num_filters = hyperparameters["num_filters"]
        self.bottleneck_dim = self.num_filters
        self.dis_hidden = self.num_filters
        self.num_classes = num_classes
        self.feature_extractor = architecture_class(num_channels, self.num_filters)
        self.classifier = nn.Linear(self.bottleneck_dim, num_classes)
        self.discriminator = Discriminator(self.bottleneck_dim, self.dis_hidden, self.num_domains)
        self.dclassifier = nn.Linear(self.bottleneck_dim, self.num_domains)
        self.ddiscriminator = Discriminator(self.bottleneck_dim, self.dis_hidden, num_classes)
        self.aclassifier = nn.Linear(self.bottleneck_dim, self.num_domains * num_classes)
        self.bottlenecks_built = False

    def build_bottlenecks(self, feat_dim, device):
        def make():
            return nn.Sequential(
                nn.Linear(feat_dim, self.bottleneck_dim),
                nn.BatchNorm1d(self.bottleneck_dim),
                nn.ReLU(),
            ).to(device)
        self.bottleneck = make()
        self.dbottleneck = make()
        self.abottleneck = make()
        self.bottlenecks_built = True

    def features(self, x):
        f = self.feature_extractor(x)
        if not self.bottlenecks_built:
            self.build_bottlenecks(f.shape[1], f.device)
        return f

    def featurize(self, x):
        f = self.features(x)
        return self.bottleneck(f)

    def classify(self, z):
        return self.classifier(z)

    def domain_classify(self, z, lambda_=1.0):
        return self.discriminator(grad_reverse(z, lambda_))

    def d_featurize(self, x):
        f = self.features(x)
        return self.dbottleneck(f)

    def d_class_logits(self, z):
        return self.dclassifier(z)

    def d_discriminate(self, z, lambda_=1.0):
        return self.ddiscriminator(grad_reverse(z, lambda_))

    def a_featurize(self, x):
        f = self.features(x)
        return self.abottleneck(f)

    def super_classify(self, z):
        return self.aclassifier(z)

    def forward(self, x):
        features = self.featurize(x)
        return {"logits": self.classify(features), "features": features}


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.layers(x)


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x, lambda_=1.0):
    return GradientReversal.apply(x, lambda_)