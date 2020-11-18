#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
# Copyright 2020 Ibuki Kuroyanagi
import argparse
import logging
import os
import sys
import warnings
import numpy as np
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
from preprocess import preprocess_pipeline
from qhoptim import QHAdam
from Tab_dataset import MoaDataset
from tab_trainer import TabTrainer

# from variables import top_feats

from utils import seed_everything

warnings.filterwarnings("ignore")


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
    parser.add_argument(
        "--dpgmmdir", type=str, default="", help="Path of dpgmm directory."
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
    # dpgmm_dir = args.outdir
    train, test = preprocess_pipeline(
        train_features,
        test_features,
        config,
        path=args.dpgmmdir,
        is_concat=config.get("is_concat", False),
    )
    logging.info(f"{train.shape}\n{train.head()}")
    logging.info(f"{test.shape}\n{test.head()}")
    drop_cols = train.columns[train.std() == 0]
    train.drop(columns=drop_cols, inplace=True)
    test.drop(columns=drop_cols, inplace=True)
    top_feats = np.arange(train.shape[1])
    drop_idx = train["cp_type"] == 0
    train = train.loc[drop_idx].reset_index(drop=True)
    del train_targets["sig_id"]
    # from IPython import embed

    # embed()
    train_targets = train_targets.loc[drop_idx].reset_index(drop=True)
    train = train.values
    test = test.values
    train_targets = train_targets.values
    logging.info("Successfully preprocessed.")
    resumes = [""]
    resumes += [
        f"{config['outdir']}/best/best_loss{fold}fold.pkl"
        for fold in range(config["n_fold"] - 1)
    ]
    logging.info(f"resumes: {resumes}")

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
        logging.info(
            f"train_set:{train_set[0]['X'].shape}, val_set:{val_set[0]['X'].shape}"
        )

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
        if len(resumes[n]) != 0:
            trainer.load_checkpoint(resumes[n])
            logging.info(f"Successfully resumed from {resumes[n]}.")

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
