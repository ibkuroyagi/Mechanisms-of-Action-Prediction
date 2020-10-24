#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Ibuki Kuroyanagi
# %%
import argparse
import logging
import sys
import os
import datetime
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append(
    "../input/modules/iterative-stratification/iterative-stratification-master"
)
sys.path.append("../input/modules/datasets")
sys.path.append("../input/modules/facebookresearch")
sys.path.append("../input/modules/trainer")
sys.path.append("../input/modules/Qwicen")
sys.path.append("../input/modules/trainer")
sys.path.append("../input/modules/utils")
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from datasets import MoaDataset
from node import NODE
from qhoptim import QHAdam
from tab_tab_trainer import TabTrainer
from utils import seed_everything
from utils import create_logger
from utils import get_logger
from utils import get_logger
from variables import top_feats

warnings.filterwarnings("ignore")


def preprocess(df):
    df = df.copy()
    df.loc[:, "cp_type"] = df.loc[:, "cp_type"].map({"trt_cp": 0, "ctl_vehicle": 1})
    df.loc[:, "cp_dose"] = df.loc[:, "cp_dose"].map({"D1": 0, "D2": 1})
    del df["sig_id"]
    return df


def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description="Train Parallel WaveGAN (See detail in parallel_wavegan/bin/train.py)."
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="Path of output directory."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path of config file."
    )
    args = parser.parse_args()

    # load and save config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))
    seed_everything(config["seed"])
    # set logger
    log_path = os.path.join(config["outdir"], "train")
    create_logger(log_path)
    get_logger(log_path).info("Successfully load config files.")
    train_features = pd.read_csv("../input/lish-moa/train_features.csv")
    train_targets = pd.read_csv("../input/lish-moa/train_targets_scored.csv")
    # train_nontargets = pd.read_csv("../input/lish-moa/train_targets_nonscored.csv")
    test_features = pd.read_csv("../input/lish-moa/test_features.csv")
    ss = pd.read_csv("../input/lish-moa/sample_submission.csv")
    get_logger(log_path).info("Successfully load input files.")
    train = preprocess(train_features)
    test = preprocess(test_features)
    del train_targets["sig_id"]
    train_targets = train_targets.loc[train["cp_type"] == 0].reset_index(drop=True)
    train = train.loc[train["cp_type"] == 0].reset_index(drop=True)

    logging.info("Successfully preprocessed.")
    nfolds = 10
    nstarts = 1
    nepochs = 200
    batch_size = 128
    val_batch_size = batch_size * 4
    ntargets = train_targets.shape[1]
    targets = [col for col in train_targets.columns]
    criterion = nn.BCELoss()

    # for GPU/CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train = train.values
    test = test.values
    train_targets = train_targets.values

    kfold = MultilabelStratifiedKFold(
        n_splits=nfolds, random_state=config["seed"], shuffle=True
    )
    for n, (tr, te) in enumerate(kfold.split(train_targets, train_targets)):
        logging.info(f"Start to train fold {nn}.")
        start_time = time.time()
        xtrain, xval = train[tr], train[te]
        ytrain, yval = train_targets[tr], train_targets[te]

        train_set = MoaDataset(xtrain, ytrain, top_feats)
        val_set = MoaDataset(xval, yval, top_feats)

        dataloaders = {
            "train": DataLoader(train_set, batch_size=batch_size, shuffle=True),
            "val": DataLoader(val_set, batch_size=val_batch_size, shuffle=False),
        }
        model = NODE(
            input_dim=len(top_feats), out_dim=206, **config["model_params"]
        ).to(device)
        checkpoint_path = os.path.join(outdir, f"Model_{seed}_Fold_{n+1}.pt")
        optimizer = QHAdam(
            model.parameters(),
            lr=1e-3,
            nus=(0.7, 1.0),
            betas=(0.95, 0.998),
            weight_decay=1e-5,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=10, eps=1e-4, verbose=False
        )
        best_loss = {"train": np.inf, "val": np.inf}

        es_count = 0
        for epoch in range(nepochs):
            epoch_loss = {"train": 0.0, "val": 0.0}

            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0

                for i, (x, y) in enumerate(dataloaders[phase]):
                    x, y = x.to(device), y.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        preds = model(x)
                        loss = criterion(preds, y)

                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() / len(dataloaders[phase])

                epoch_loss[phase] = running_loss

            scheduler.step(epoch_loss["val"])

            if epoch_loss["val"] < best_loss["val"]:
                best_loss = epoch_loss
                torch.save(model.state_dict(), checkpoint_path)
                es_count = 0
            else:
                es_count += 1

            print(
                "Epoch {}/{} - loss: {:5.5f} - val_loss: {:5.5f} - es: {}".format(
                    epoch + 1, nepochs, epoch_loss["train"], epoch_loss["val"], es_count
                )
            )

            if es_count > 20:
                break

        print(
            "[{}] - seed: {} - fold: {} - best val_loss: {:5.5f}".format(
                str(datetime.timedelta(seconds=time.time() - start_time))[2:7],
                seed,
                n,
                best_loss["val"],
            )
        )

    # %%
    oof = np.zeros((len(train), nstarts, ntargets))
    oof_targets = np.zeros((len(train), ntargets))
    preds = np.zeros((len(test), ntargets))

    # %%

    print(f"Inference for seed {seed}")
    seed_targets = []
    seed_oof = []
    seed_preds = np.zeros((len(test), ntargets, nfolds))

    for n, (tr, te) in enumerate(kfold.split(train_targets, train_targets)):
        xval, yval = train[te], train_targets[te]
        fold_preds = []

        val_set = MoaDataset(xval, yval, top_feats)
        test_set = MoaDataset(test, None, top_feats, mode="test")

        dataloaders = {
            "val": DataLoader(val_set, batch_size=val_batch_size, shuffle=False),
            "test": DataLoader(test_set, batch_size=val_batch_size, shuffle=False),
        }

        checkpoint_path = os.path.join(outdir, f"Model_{seed}_Fold_{n+1}.pt")
        model = NODE(
            input_dim=len(top_feats), out_dim=206, **config["model_params"]
        ).to(device)
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()

        for phase in ["val", "test"]:
            for i, (x, y) in enumerate(dataloaders[phase]):
                if phase == "val":
                    x, y = x.to(device), y.to(device)
                elif phase == "test":
                    x = x.to(device)

                with torch.no_grad():
                    batch_preds = model(x)

                    if phase == "val":
                        seed_targets.append(y)
                        seed_oof.append(batch_preds)
                    elif phase == "test":
                        fold_preds.append(batch_preds)

        fold_preds = torch.cat(fold_preds, dim=0).cpu().numpy()
        seed_preds[:, :, n] = fold_preds

    seed_targets = torch.cat(seed_targets, dim=0).cpu().numpy()
    seed_oof = torch.cat(seed_oof, dim=0).cpu().numpy()
    seed_preds = np.mean(seed_preds, axis=2)

    print("Score for this seed {:5.5f}".format(mean_log_loss(seed_targets, seed_oof)))
    oof_targets = seed_targets
    oof[:, seed, :] = seed_oof
    preds += seed_preds / nstarts

    oof = np.mean(oof, axis=1)
    print("Overall score is {:5.5f}".format(mean_log_loss(oof_targets, oof)))

    ss[targets] = preds
    ss.loc[test_features["cp_type"] == "ctl_vehicle", targets] = 0
    ss.to_csv("submission_tmp.csv", index=False)
