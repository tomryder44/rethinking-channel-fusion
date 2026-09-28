# import numpy as np
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
#
#
# def compute_pad(source_features, target_features):
#
#     sf = source_features.detach().cpu().numpy()
#     tf = target_features.detach().cpu().numpy()
#
#     sl = np.zeros(sf.shape[0])
#     tl = np.ones(tf.shape[0])
#
#     sf_tr, sf_te, sl_tr, sl_te = train_test_split(sf, sl, test_size=0.25)
#     tf_tr, tf_te, tl_tr, tl_te = train_test_split(tf, tl, test_size=0.25)
#
#     x_train = np.concatenate([sf_tr, tf_tr], axis=0)
#     y_train = np.concatenate([sl_tr, tl_tr], axis=0)
#
#     x_test = np.concatenate([sf_te, tf_te], axis=0)
#     y_test = np.concatenate([sl_te, tl_te], axis=0)
#
#     clf = LogisticRegression(class_weight="balanced", max_iter=1000)
#     clf.fit(x_train, y_train)
#
#     preds = clf.predict(x_test)
#
#     err_s = (preds[y_test == 0] != 0).mean()
#     err_t = (preds[y_test == 1] != 1).mean()
#     test_error = 0.5 * (err_s + err_t)
#
#     return max(0.0, 2.0 * (1.0 - 2.0 * test_error))


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def compute_pad_(source_features, target_features, n_splits=10, seed=0, C=1.0):
    sf = source_features.detach().cpu().numpy()
    tf = target_features.detach().cpu().numpy()

    # use all samples from both sides; class_weight handles the imbalance
    X = np.concatenate([sf, tf], axis=0)
    y = np.concatenate([np.zeros(len(sf)), np.ones(len(tf))])

    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, class_weight="balanced", max_iter=5000),
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    errs = []
    for tr, te in cv.split(X, y):
        pipe.fit(X[tr], y[tr])
        p, yt = pipe.predict(X[te]), y[te]
        errs.append(0.5 * ((p[yt == 0] != 0).mean() + (p[yt == 1] != 1).mean()))

    return max(0.0, 2.0 * (1.0 - 2.0 * np.mean(errs)))


def compute_pad(source_features, target_features, n_boot=50):
    pads = [compute_pad_(source_features, target_features, seed=s) for s in range(n_boot)]
    return float(np.mean(pads))
