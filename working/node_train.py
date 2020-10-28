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
    # ntargets = train_targets.shape[1]
    # targets = [col for col in train_targets.columns]
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
            trainer.run()
        except KeyboardInterrupt:
            trainer.save_checkpoint(
                os.path.join(config["outdir"], f"checkpoint-{trainer.steps}steps.pkl")
            )
            logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")
    # oof = np.zeros((len(train), nstarts, ntargets))
    # oof_targets = np.zeros((len(train), ntargets))
    # preds = np.zeros((len(test), ntargets))

    # # %%

    # print(f"Inference for seed {seed}")
    # seed_targets = []
    # seed_oof = []
    # seed_preds = np.zeros((len(test), ntargets, nfolds))

    # for n, (tr, te) in enumerate(kfold.split(train_targets, train_targets)):
    #     xval, yval = train[te], train_targets[te]
    #     fold_preds = []

    #     val_set = MoaDataset(xval, yval, top_feats)
    #     test_set = MoaDataset(test, None, top_feats, mode="test")

    #     dataloaders = {
    #         "val": DataLoader(val_set, batch_size=val_batch_size, shuffle=False),
    #         "test": DataLoader(test_set, batch_size=val_batch_size, shuffle=False),
    #     }

    #     checkpoint_path = os.path.join(outdir, f"Model_{seed}_Fold_{n+1}.pt")
    #     model = NODE(
    #         input_dim=len(top_feats), out_dim=206, **config["model_params"]
    #     ).to(device)
    #     model.load_state_dict(torch.load(checkpoint_path))
    #     model.eval()

    #     for phase in ["val", "test"]:
    #         for i, (x, y) in enumerate(dataloaders[phase]):
    #             if phase == "val":
    #                 x, y = x.to(device), y.to(device)
    #             elif phase == "test":
    #                 x = x.to(device)

    #             with torch.no_grad():
    #                 batch_preds = model(x)

    #                 if phase == "val":
    #                     seed_targets.append(y)
    #                     seed_oof.append(batch_preds)
    #                 elif phase == "test":
    #                     fold_preds.append(batch_preds)

    #     fold_preds = torch.cat(fold_preds, dim=0).cpu().numpy()
    #     seed_preds[:, :, n] = fold_preds

    # seed_targets = torch.cat(seed_targets, dim=0).cpu().numpy()
    # seed_oof = torch.cat(seed_oof, dim=0).cpu().numpy()
    # seed_preds = np.mean(seed_preds, axis=2)

    # print("Score for this seed {:5.5f}".format(mean_log_loss(seed_targets, seed_oof)))
    # oof_targets = seed_targets
    # oof[:, seed, :] = seed_oof
    # preds += seed_preds / nstarts

    # oof = np.mean(oof, axis=1)
    # print("Overall score is {:5.5f}".format(mean_log_loss(oof_targets, oof)))

    # ss[targets] = preds
    # ss.loc[test_features["cp_type"] == "ctl_vehicle", targets] = 0
    # ss.to_csv("submission_tmp.csv", index=False)


if __name__ == "__main__":
    main()
