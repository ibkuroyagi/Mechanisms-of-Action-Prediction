#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Ibuki Kuroyanagi
import argparse
import logging
import os
import sys
import warnings

import pandas as pd
import torch
import yaml
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