from geomloss import SamplesLoss


def compute_gaussian_mmd(source_features, target_features, bw):
    loss = SamplesLoss(loss="gaussian", blur=bw)
    l = loss(source_features, target_features).item()
    return l


def compute_energy_mmd(source_features, target_features):
    loss = SamplesLoss(loss="energy")
    return loss(source_features, target_features).item()
