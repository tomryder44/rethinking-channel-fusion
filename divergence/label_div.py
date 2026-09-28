import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


def compute_optimal_joint_error(
        source_features,
        source_labels,
        source_domains,
        target_features,
        target_labels
):

    device = source_features.device
    batch_size = 32

    source_features = source_features.detach().cpu()
    source_labels = source_labels.detach().cpu().long()
    source_domains = source_domains.detach().cpu()
    target_features = target_features.detach().cpu()
    target_labels = target_labels.detach().cpu().long()

    all_source_domains = torch.unique(source_domains)
    x_train, y_train, source_x_test, source_y_test = [], [], [], []
    for d in all_source_domains:
        mask_d = (source_domains == d)
        source_features_d = source_features[mask_d]
        source_labels_d = source_labels[mask_d]

        v, c = torch.unique(source_labels_d, return_counts=True)
        keep = torch.isin(source_labels_d, v[c >= 2])
        source_features_d = source_features_d[keep]
        source_labels_d = source_labels_d[keep]

        sf_tr, sf_te, sl_tr, sl_te = train_test_split(source_features_d,
                                                      source_labels_d,
                                                      test_size=0.25,
                                                      stratify=source_labels_d.numpy())

        x_train.append(sf_tr)
        y_train.append(sl_tr)
        source_x_test.append(sf_te)
        source_y_test.append(sl_te)

    source_x_test = torch.cat(source_x_test, dim=0)
    source_y_test = torch.cat(source_y_test, dim=0).long()

    v, c = torch.unique(target_labels, return_counts=True)
    keep = torch.isin(target_labels, v[c >= 2])
    target_features = target_features[keep]
    target_labels = target_labels[keep]

    tf_tr, tf_te, tl_tr, tl_te = train_test_split(target_features,
                                                  target_labels,
                                                  test_size=0.25,
                                                  stratify=target_labels.numpy())

    x_train.append(tf_tr)
    y_train.append(tl_tr)

    target_x_test = tf_te
    target_y_test = tl_te

    #

    x_train = torch.cat(x_train, dim=0)
    y_train = torch.cat(y_train, dim=0).long()

    train_ds = TensorDataset(x_train.to(device), y_train.to(device))
    s_test_ds = TensorDataset(source_x_test.to(device), source_y_test.to(device))
    t_test_ds = TensorDataset(target_x_test.to(device), target_y_test.to(device))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    s_test_loader = DataLoader(s_test_ds, batch_size=batch_size, shuffle=False)
    t_test_loader = DataLoader(t_test_ds, batch_size=batch_size, shuffle=False)

    num_classes = int(y_train.unique().numel())

    weights = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

    best = 1e10

    for weight in weights:

        model = nn.Linear(x_train.shape[1], num_classes).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight)

        model.train()
        max_epochs = 500
        best_loss = 1e10
        thr = 1e-4
        patience = 10
        for epoch in range(max_epochs):
            epoch_train_err = 0.
            for xb, yb in train_loader:
                with torch.enable_grad():
                    opt.zero_grad()
                    out = model(xb)
                    loss = F.cross_entropy(out, yb)
                    loss.backward()
                    opt.step()
                    epoch_train_err += loss.item()
            epoch_train_err /= len(train_loader)
            improvement = best_loss - epoch_train_err
            if improvement > thr:
                best_loss = epoch_train_err
                patience = 10
            else:
                patience -= 1
            if patience == 0:
                break

        model.eval()

        logits = []
        labels = []
        for xb, yb in s_test_loader:
            out = model(xb)
            logits.append(out)
            labels.append(yb)
        s_logits = torch.cat(logits)
        s_labels = torch.cat(labels)
        s_loss = F.cross_entropy(s_logits, s_labels).item()
        s_err = (s_logits.argmax(1) != s_labels).sum().item()

        logits = []
        labels = []
        for xb, yb in t_test_loader:
            out = model(xb)
            logits.append(out)
            labels.append(yb)
        t_logits = torch.cat(logits)
        t_labels = torch.cat(labels)
        t_loss = F.cross_entropy(t_logits, t_labels).item()
        t_err = (t_logits.argmax(1) != t_labels).sum().item()

        if s_err + t_err < best:
            best = s_err + t_err
            best_l = (s_loss + t_loss) / 2  # AVERAGE

    # return {"loss": float(s_loss + t_loss), "loss_s": s_loss, "loss_t": t_loss,
    #         "error": float(s_err + t_err), "error_s": s_err, "error_t": t_err}

    return {"error": float(best), "loss": float(best_l)}


































def compute_optimal_joint_error_ensemble(
        source_features,
        source_labels,
        source_domains,
        target_features,
        target_labels
):

    device = source_features.device
    batch_size = 32

    sf = source_features.detach().cpu()
    sl = source_labels.detach().cpu().long()
    sd = source_domains.detach().cpu()

    tf = target_features.detach().cpu()
    tl = target_labels.detach().cpu().long()

    domains = torch.unique(sd)
    x_train, y_train, xs_test, ys_test = [], [], [], []

    sf = torch.permute(sf, (1, 0, 2))
    tf = torch.permute(tf, (1, 0, 2))

    for d in domains:
        mask_d = (sd == d)
        sf_d = sf[mask_d]
        sl_d = sl[mask_d]

        # keep only classes with >=2 samples for stratification
        v, c = torch.unique(sl_d, return_counts=True)
        keep = torch.isin(sl_d, v[c >= 2])
        sf_d = sf_d[keep]
        sl_d = sl_d[keep]

        sf_tr, sf_te, sl_tr, sl_te = train_test_split(sf_d, sl_d, test_size=0.25, stratify=sl_d.numpy())

        x_train.append(sf_tr)
        y_train.append(sl_tr)
        xs_test.append(sf_te)
        ys_test.append(sl_te)

    tf_tr, tf_te, tl_tr, tl_te = train_test_split(tf, tl, test_size=0.25, stratify=tl.numpy())

    x_train.append(tf_tr)
    y_train.append(tl_tr)

    x_train = torch.cat(x_train, dim=0)
    y_train = torch.cat(y_train, dim=0).long()

    xs_test = torch.cat(xs_test, dim=0)
    ys_test = torch.cat(ys_test, dim=0).long()

    tf_test = tf_te
    ty_test = tl_te

    train_ds = TensorDataset(x_train.to(device), y_train.to(device))
    s_test_ds = TensorDataset(xs_test.to(device), ys_test.to(device))
    t_test_ds = TensorDataset(tf_test.to(device), ty_test.to(device))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    s_test_loader = DataLoader(s_test_ds, batch_size=batch_size, shuffle=False)
    t_test_loader = DataLoader(t_test_ds, batch_size=batch_size, shuffle=False)

    num_classes = int(y_train.unique().numel())
    num_channels = x_train.shape[1]
    model = nn.ModuleList([nn.Linear(x_train.shape[2], num_classes) for _ in range(num_channels)]).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    model.train()
    max_epochs = 500
    best_loss = 1e10
    thr = 1e-2
    patience = 10
    for epoch in range(max_epochs):
        epoch_train_err = 0.
        for xb, yb in train_loader:
            loss = 0.
            with torch.enable_grad():
                opt.zero_grad(set_to_none=True)
                for i in range(num_channels):
                    x_c = xb[:, i]
                    out_c = model[i](x_c)
                    loss_c = F.cross_entropy(out_c, yb)
                    loss += loss_c
                loss.backward()
                opt.step()
            ens_pred = torch.mean(torch.stack([model[i](xb[:, i]) for i in range(num_channels)]), dim=0)
            ens_loss = F.cross_entropy(ens_pred, yb)
            epoch_train_err += ens_loss.item()
        epoch_train_err /= len(train_loader)
        # print(epoch, epoch_train_err)
        improvement = best_loss - epoch_train_err
        if improvement > thr:
            best_loss = epoch_train_err
            patience = 10
        else:
            patience -= 1
        if patience == 0:
            break

    model.eval()
    s_loss = 0.
    s_err = 0.
    n_samples = 0
    for xb, yb in s_test_loader:
        out = torch.mean(torch.stack([model[i](xb[:, i]) for i in range(num_channels)]), dim=0)
        s_loss += F.cross_entropy(out, yb, reduction="sum").item()
        s_err += (out.argmax(1) != yb).sum().item()
        n_samples += yb.shape[0]
    s_loss /= n_samples
    s_err /= n_samples

    t_loss = 0.
    t_err = 0.
    n_samples = 0
    for xb, yb in t_test_loader:
        out = torch.mean(torch.stack([model[i](xb[:, i]) for i in range(num_channels)]), dim=0)
        t_loss += F.cross_entropy(out, yb, reduction="sum").item()
        t_err += (out.argmax(1) != yb).sum().item()
        n_samples += yb.shape[0]
    t_loss /= n_samples
    t_err /= n_samples

    return {"loss": float(s_loss + t_loss), "loss_s": s_loss, "loss_t": t_loss,
            "error": float(s_err + t_err), "error_s": s_err, "error_t": t_err}

