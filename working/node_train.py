#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
# Copyright 2020 Ibuki Kuroyanagi
import argparse
import logging
import os
import sys
import warnings

import pandas as pd
import torch
import yaml
from sklearn.cluster import KMeans
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

sys.path.append("../input/iterative-stratification/iterative-stratification-master")
sys.path.append("../input/modules/datasets")
sys.path.append("../input/modules/facebookresearch")
sys.path.append("../input/modules/losses")
sys.path.append("../input/modules/Qwicen")
sys.path.append("../input/modules/trainer")
sys.path.append("../input/modules/utils")
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from label_smooth_loss import SmoothBCEwLogits
from node import NODE
from qhoptim import QHAdam
from Tab_dataset import MoaDataset
from tab_trainer import TabTrainer
from variables import top_feats

from utils import seed_everything

warnings.filterwarnings("ignore")


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
        transformer = QuantileTransformer(**config["QuantileTransformer"])
        vec_len = len(train_features[col].values)
        vec_len_test = len(test_features[col].values)
        raw_vec = train_features[col].values.reshape(vec_len, 1)
        transformer.fit(raw_vec)
        train_features[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]
        test_features[col] = transformer.transform(
            test_features[col].values.reshape(vec_len_test, 1)
        ).reshape(1, vec_len_test)[0]
        return train_features, test_features


def apply_pca(train_features, test_features, columns, n_comp=35, kind="g", SEED=42):
    data = pd.concat(
        [pd.DataFrame(train_features[columns]), pd.DataFrame(test_features[columns])]
    )
    data2 = PCA(n_components=n_comp, random_state=SEED).fit_transform(data[columns])
    train2 = data2[: train_features.shape[0]]
    test2 = data2[-test_features.shape[0] :]

    train2 = pd.DataFrame(train2, columns=[f"pca_{kind}-{i}" for i in range(n_comp)])
    test2 = pd.DataFrame(test2, columns=[f"pca_{kind}-{i}" for i in range(n_comp)])
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


def preprocess_pipeline(train_features, test_features, config):
    GENES = [col for col in train_features.columns if col.startswith("g-")]
    CELLS = [col for col in train_features.columns if col.startswith("c-")]
    # original statics
    train_features, test_features = fe_stats(
        train_features, test_features, GENES, kind="g"
    )
    train_features, test_features = fe_stats(
        train_features, test_features, CELLS, kind="c"
    )
    # RankGauss
    train_features, test_features = apply_rank_gauss(
        train_features, test_features, columns=GENES + CELLS, config=config
    )
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
    print(statics_cols)
    train_features, test_features = apply_zscore(
        train_features, test_features, statics_cols
    )
    # PCA
    train_features, test_features = apply_pca(
        train_features,
        test_features,
        columns=GENES,
        n_comp=config["n_comp_g"],
        kind="g",
        SEED=config["seed"],
    )
    train_features, test_features = apply_pca(
        train_features,
        test_features,
        columns=CELLS,
        n_comp=config["n_comp_c"],
        kind="c",
        SEED=config["seed"],
    )
    # Variance Threshold
    if config.get("VarianceThreshold", 0) != 0:
        train_features, test_features = reduce_columns(
            train_features, test_features, threshold=config["VarianceThreshold"]
        )
    # k-means++
    train_features, test_features = create_cluster(
        train_features,
        test_features,
        GENES,
        n_clusters=config["n_cluster_g"],
        SEED=config["seed"],
        kind="g",
    )
    train_features, test_features = create_cluster(
        train_features,
        test_features,
        CELLS,
        n_clusters=config["n_cluster_c"],
        SEED=config["seed"],
        kind="c",
    )
    train = preprocess(train_features)
    train = train.loc[train["cp_type"] == 0].reset_index(drop=True)
    test = preprocess(test_features)
    # return train, test
    return train.values, test.values


# %%
def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description="Train Parallel WaveGAN (See detail in parallel_wavegan/bin/train.py)."
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="Path of output directory."
    )
    parser.add_argument(
        "--resume", type=str, default="", help="Path of resumed model file."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path of config file."
    )
    parser.add_argument("--verbose", type=int, default=1, help="verbose")
    args = parser.parse_args()
    # load and save config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))
    # global setting
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_everything(config["seed"])
    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")
    for key, value in config.items():
        logging.info(f"{key} = {value}")
    # preprocess
    train_features = pd.read_csv("../input/lish-moa/train_features.csv")
    train_targets = pd.read_csv("../input/lish-moa/train_targets_scored.csv")
    train_targets = train_targets.loc[train["cp_type"] == 0].reset_index(drop=True)
    # train_nontargets = pd.read_csv("../input/lish-moa/train_targets_nonscored.csv")
    test_features = pd.read_csv("../input/lish-moa/test_features.csv")
    logging.info("Successfully load input files.")
    train, test = preprocess_pipeline(train_features, test_features, config)
    del train_targets["sig_id"]
    train_targets = train_targets.values
    logging.info("Successfully preprocessed.")
    if config.get("loss_type", "BCELoss") == "SmoothBCEwLogits":
        loss_class = SmoothBCEwLogits
    else:
        loss_class = getattr(
            torch.nn,
            # keep compatibility
            config.get("loss_type", "BCELoss"),
        )
    criterion = loss_class(**config["loss_params"]).to(device)

    # for GPU/CPU
    kfold = MultilabelStratifiedKFold(
        n_splits=config["n_fold"], random_state=config["seed"], shuffle=True
    )
    for n, (tr, te) in enumerate(kfold.split(train_targets, train_targets)):
        logging.info(f"Start to train fold {n}.")
        xtrain, xval = train[tr], train[te]
        ytrain, yval = train_targets[tr], train_targets[te]

        train_set = MoaDataset(xtrain, ytrain, top_feats)
        val_set = MoaDataset(xval, yval, top_feats)

        data_loader = {
            "train": DataLoader(
                train_set,
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
                pin_memory=config["pin_memory"],
                shuffle=True,
            ),
            "dev": DataLoader(
                val_set,
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
                pin_memory=config["pin_memory"],
                shuffle=False,
            ),
        }
        model = NODE(
            input_dim=len(top_feats),
            out_dim=config["out_dim"],
            **config["model_params"],
        ).to(device)

        if config["optimizer_type"] == "QHAdam":
            optimizer_class = QHAdam
        else:
            optimizer_class = getattr(
                torch.optim,
                # keep compatibility
                config.get("optimizer_type", "Adam"),
            )
        optimizer = optimizer_class(
            params=model.parameters(), **config["optimizer_params"]
        )

        scheduler_class = getattr(
            torch.optim.lr_scheduler,
            # keep compatibility
            config.get("scheduler_type", "StepLR"),
        )
        scheduler = scheduler_class(optimizer=optimizer, **config["scheduler_params"])
        trainer = TabTrainer(
            steps=0,
            epochs=0,
            data_loader=data_loader,
            model=model.to(device),
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device,
            add_name=f"{n}fold",
            train=True,
        )
        # resume from checkpoint
        if len(args.resume) != 0:
            trainer.load_checkpoint(args.resume)
            logging.info(f"Successfully resumed from {args.resume}.")

        # run training loop
        try:
            logging.info("Start training!")
            trainer.run()
        except KeyboardInterrupt:
            trainer.save_checkpoint(
                os.path.join(config["outdir"], f"checkpoint-{trainer.steps}steps.pkl")
            )
            logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")


if __name__ == "__main__":
    main()
