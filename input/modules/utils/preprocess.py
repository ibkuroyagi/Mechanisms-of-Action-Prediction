import joblib
import os
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def apply_zscore(train_features, test_features, columns):
    for col in columns:
        transformer = StandardScaler()
        vec_len = len(train_features[col].values)
        vec_len_test = len(test_features[col].values)
        raw_vec = train_features[col].values.reshape(vec_len, 1)
        transformer.fit(raw_vec)
        train_features[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]
        test_features[col] = transformer.transform(
            test_features[col].values.reshape(vec_len_test, 1)
        ).reshape(1, vec_len_test)[0]
    return train_features, test_features


def apply_rank_gauss(train_features, test_features, columns, config):
    for col in columns:
        transformer = QuantileTransformer(**config)
        vec_len = len(train_features[col].values)
        vec_len_test = len(test_features[col].values)
        raw_vec = train_features[col].values.reshape(vec_len, 1)
        transformer.fit(raw_vec)
        train_features[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]
        test_features[col] = transformer.transform(
            test_features[col].values.reshape(vec_len_test, 1)
        ).reshape(1, vec_len_test)[0]
    return train_features, test_features


def apply_pca(train_features, test_features, columns, threshold=0.9, kind="g", SEED=42):
    pca = PCA(random_state=SEED)
    pca.fit(train_features[columns])
    train2 = pca.transform(train_features[columns])
    test2 = pca.transform(test_features[columns])
    last_idx = int(sum(pca.explained_variance_ratio_.cumsum() < threshold))
    train2 = pd.DataFrame(
        train2[:, :last_idx], columns=[f"pca_{kind}-{i}" for i in range(last_idx)]
    )
    test2 = pd.DataFrame(
        test2[:, :last_idx], columns=[f"pca_{kind}-{i}" for i in range(last_idx)]
    )
    train_features = pd.concat((train_features, train2), axis=1)
    test_features = pd.concat((test_features, test2), axis=1)
    return train_features, test_features


def reduce_columns(train_features, test_features, threshold=0.8):
    from sklearn.feature_selection import VarianceThreshold

    var_thresh = VarianceThreshold(threshold)
    data = train_features.append(test_features)
    var_thresh.fit_transform(data.iloc[:, 4:])

    train_features = data.loc[: train_features.shape[0], var_thresh.get_support()]
    test_features = data.loc[-test_features.shape[0] :, var_thresh.get_support()]
    return train_features, test_features


def create_cluster(
    train,
    test,
    features,
    n_clusters=35,
    SEED=42,
    kind="g",
):
    train_ = train[features].copy()
    test_ = test[features].copy()
    data = pd.concat([train_, test_], axis=0)
    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED).fit(data)
    train[f"clusters_{kind}"] = kmeans.labels_[: train.shape[0]]
    test[f"clusters_{kind}"] = kmeans.labels_[train.shape[0] :]
    train = pd.get_dummies(train, columns=[f"clusters_{kind}"])
    test = pd.get_dummies(test, columns=[f"clusters_{kind}"])
    return train, test


def create_dpgmm_proba(
    train_features, test_features, columns, path=None, config={}, kind="g"
):
    from sklearn.mixture import BayesianGaussianMixture

    if path is None:
        dpgmm = BayesianGaussianMixture(**config)
        dpgmm.fit(train_features[columns])
    else:
        with open(path, "rb") as f:
            dpgmm = joblib.load(f)
    train2 = dpgmm.predict_proba(train_features[columns])
    test2 = dpgmm.predict_proba(test_features[columns])
    n_cluster = train2.shape[1]
    train2 = pd.DataFrame(
        train2, columns=[f"dpgmm_{kind}-{i}" for i in range(n_cluster)]
    )
    test2 = pd.DataFrame(test2, columns=[f"dpgmm_{kind}-{i}" for i in range(n_cluster)])
    train_features = pd.concat((train_features, train2), axis=1)
    test_features = pd.concat((test_features, test2), axis=1)
    return train_features, test_features


def fe_stats(train, test, columns, kind="g"):
    for df in train, test:
        df[f"{kind}_sum"] = df[columns].sum(axis=1)
        df[f"{kind}_mean"] = df[columns].mean(axis=1)
        df[f"{kind}_std"] = df[columns].std(axis=1)
        df[f"{kind}_kurt"] = df[columns].kurtosis(axis=1)
        df[f"{kind}_skew"] = df[columns].skew(axis=1)
    return train, test


def preprocess(df):
    df = df.copy()
    df.loc[:, "cp_type"] = df.loc[:, "cp_type"].map({"trt_cp": 0, "ctl_vehicle": 1})
    df.loc[:, "cp_dose"] = df.loc[:, "cp_dose"].map({"D1": 0, "D2": 1})
    df.loc[:, "cp_time"] = df.loc[:, "cp_time"] / 72.0
    del df["sig_id"]
    return df


def preprocess_pipeline(train_features, test_features, config, path=None):
    GENES = [col for col in train_features.columns if col.startswith("g-")]
    CELLS = [col for col in train_features.columns if col.startswith("c-")]
    # original statics
    train_features, test_features = fe_stats(
        train_features, test_features, GENES, kind="g"
    )
    train_features, test_features = fe_stats(
        train_features, test_features, CELLS, kind="c"
    )
    print("Successfully caluculate original statistics.")

    if config["norm_type"] == "zscore":
        # zscore
        train_features, test_features = apply_zscore(
            train_features,
            test_features,
            columns=GENES + CELLS,
        )
    elif config["norm_type"] == "RankGauss":
        # RankGauss
        train_features, test_features = apply_rank_gauss(
            train_features,
            test_features,
            columns=GENES + CELLS,
            config=config["QuantileTransformer"],
        )
    print(f"Successfully caluculate {config['norm_type']}.")
    # normalized statics
    train_features, test_features = fe_stats(
        train_features, test_features, GENES, kind="norm_g"
    )
    train_features, test_features = fe_stats(
        train_features, test_features, CELLS, kind="norm_c"
    )
    statics_cols = [
        col
        for col in train_features.columns
        if col.endswith(("sum", "mean", "std", "kurt", "skew"))
    ]
    train_features, test_features = apply_zscore(
        train_features, test_features, statics_cols
    )
    print(f"Successfully caluculate {config['norm_type']} statistics.")
    # PCA
    train_features, test_features = apply_pca(
        train_features,
        test_features,
        columns=GENES,
        threshold=config["pca_threshold"],
        kind="g",
        SEED=config["seed"],
    )
    train_features, test_features = apply_pca(
        train_features,
        test_features,
        columns=CELLS,
        threshold=config["pca_threshold"],
        kind="c",
        SEED=config["seed"],
    )
    pca_cols = [col for col in train_features.columns if col.startswith("pca")]
    train_features, test_features = apply_zscore(
        train_features, test_features, pca_cols
    )
    print("Successfully caluculate PCA.")
    # Variance Threshold
    if config.get("VarianceThreshold", 0) != 0:
        train_features, test_features = reduce_columns(
            train_features, test_features, threshold=config["VarianceThreshold"]
        )
        print("Successfully caluculate Variance Threshold.")
    # k-means++
    # train_features, test_features = create_cluster(
    #     train_features,
    #     test_features,
    #     GENES,
    #     n_clusters=config["n_cluster_g"],
    #     SEED=config["seed"],
    #     kind="g",
    # )
    # train_features, test_features = create_cluster(
    #     train_features,
    #     test_features,
    #     CELLS,
    #     n_clusters=config["n_cluster_c"],
    #     SEED=config["seed"],
    #     kind="c",
    # )
    # print("Successfully caluculate k-means++.")
    if path is None:
        dpgmm_path_g = None
        dpgmm_path_c = None
    else:
        dpgmm_path_g = os.path.join(path, f"dpgmm_{config['norm_type']}_g.job")
        dpgmm_path_c = os.path.join(path, f"dpgmm_{config['norm_type']}_c.job")
    if config.get("BayesianGaussianMixture_g", None) is not None:
        train_features, test_features = create_dpgmm_proba(
            train_features,
            test_features,
            GENES,
            path=dpgmm_path_g,
            config=config["BayesianGaussianMixture_g"],
            kind="g",
        )
        print("Successfully caluculate dpgmm-g.")
    if config.get("BayesianGaussianMixture_c", None) is not None:
        train_features, test_features = create_dpgmm_proba(
            train_features,
            test_features,
            CELLS,
            path=dpgmm_path_c,
            config=config["BayesianGaussianMixture_c"],
            kind="c",
        )
        print("Successfully caluculate dpgmm-c.")
    train = preprocess(train_features)
    test = preprocess(test_features)
    print("Successfully decode categorical features.")
    return train, test
    # return train.values, test.values
