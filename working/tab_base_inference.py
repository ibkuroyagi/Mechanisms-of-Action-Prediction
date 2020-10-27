#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Ibuki Kuroyanagi
# %%
import argparse
import logging
import sys
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.append(
    "../input/modules/iterative-stratification/iterative-stratification-master"
)
sys.path.append("../input/modules/datasets")
sys.path.append("../input/modules/facebookresearch")
sys.path.append("../input/modules/Qwicen")
sys.path.append("../input/modules/trainer")
sys.path.append("../input/modules/utils")
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from Tab_dataset import MoaDataset
from node import NODE
from qhoptim import QHAdam
from tab_trainer import mean_log_loss
from tab_trainer import TabTrainer
from utils import seed_everything
from variables import top_feats

warnings.filterwarnings("ignore")


def preprocess(df):
    df = df.copy()
    df.loc[:, "cp_type"] = df.loc[:, "cp_type"].map({"trt_cp": 0, "ctl_vehicle": 1})
    df.loc[:, "cp_dose"] = df.loc[:, "cp_dose"].map({"D1": 0, "D2": 1})
    del df["sig_id"]
    return df


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
    # train_nontargets = pd.read_csv("../input/lish-moa/train_targets_nonscored.csv")
    test_features = pd.read_csv("../input/lish-moa/test_features.csv")
    logging.info("Successfully load input files.")
    train = preprocess(train_features)
    test = preprocess(test_features)
    del train_targets["sig_id"]
    train_targets = train_targets.loc[train["cp_type"] == 0].reset_index(drop=True)
    train = train.loc[train["cp_type"] == 0].reset_index(drop=True)
    train = train.values
    test = test.values
    train_targets = train_targets.values
    ntargets = train_targets.shape[1]
    targets = [col for col in train_targets.columns]
    logging.info("Successfully preprocessed.")

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
    eval_set = MoaDataset(test, None, top_feats, mode="test")
    eval_loader = {
        "eval": DataLoader(
            eval_set,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
            shuffle=False,
        ),
    }
    oof_targets = np.zeros((len(train), ntargets))
    preds = np.zeros((config["n_fold"], len(test), ntargets))
    for n, (tr, te) in enumerate(kfold.split(train_targets, train_targets)):
        logging.info(f"Start to train fold {n}.")
        xval = train[te]
        yval = train_targets[te]
        dev_set = MoaDataset(xval, yval, top_feats)
        dev_loader = {
            "eval": DataLoader(
                dev_set,
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
        # develop data
        trainer = TabTrainer(
            steps=0,
            epochs=0,
            data_loader=dev_loader,
            model=model.to(device),
            criterion=criterion,
            optimizer={},
            scheduler={},
            config=config,
            device=device,
            add_name=f"{n}fold",
        )
        trainer.load_checkpoint(args.checkpoint[n])
        logging.info(f"Successfully load checkpoint from {args.resume}.")
        oof_targets[te] = trainer.inference()
        logging.info(f"Successfully inference dev data at fold{n}.")
        fold_score = mean_log_loss(oof_targets[te], yval)
        logging.info(f"fold{n} score: {fold_score:.5f}.")
        # eval data
        trainer = TabTrainer(
            steps=0,
            epochs=0,
            data_loader=eval_loader,
            model=model.to(device),
            criterion=criterion,
            optimizer={},
            scheduler={},
            config=config,
            device=device,
            add_name=f"{n}fold",
        )
        trainer.load_checkpoint(args.checkpoint[n])
        logging.info(f"Successfully load checkpoint from {args.resume}.")
        # run training loop
        preds[n] = trainer.inference()
        logging.info(f"Successfully inference eval data at fold{n}.")
    cv_score = mean_log_loss(oof_targets, train_targets)
    logging.info(f"CV score: {cv_score:.5f}")
    train_targets = pd.read_csv("../input/lish-moa/train_targets_scored.csv")
    train_targets[targets] = oof_targets
    train_targets.to_csv("oof.csv", index=False)
    preds_mean = preds.mean(axis=0)
    ss = pd.read_csv("../input/lish-moa/sample_submission.csv")
    ss[targets] = preds_mean
    ss.loc[test_features["cp_type"] == "ctl_vehicle", targets] = 0
    ss.to_csv("submission_tmp.csv", index=False)


if __name__ == "__main__":
    main()
