import argparse
import joblib
import logging
import sys
import os
import yaml
from sklearn.mixture import BayesianGaussianMixture
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append("../input/modules/utils")

from preprocess import apply_zscore
from preprocess import apply_rank_gauss


def main():
    parser = argparse.ArgumentParser(
        description="Train Parallel WaveGAN (See detail in parallel_wavegan/bin/train.py)."
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="Path of output directory."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path of config file."
    )
    parser.add_argument("--verbose", type=int, default=1, help="verbose")
    args = parser.parse_args()
    logging.info("get args")
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
    # load and save config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))
    for key, value in config.items():
        logging.info(f"{key} = {value}")
    train_features = pd.read_csv("../input/lish-moa/train_features.csv")
    test_features = pd.read_csv("../input/lish-moa/test_features.csv")
    GENES = [col for col in train_features.columns if col.startswith("g-")]
    CELLS = [col for col in train_features.columns if col.startswith("c-")]
    logging.info("load data.")

    if config["norm_type"] == "RankGauss":
        train_features, test_features = apply_rank_gauss(
            train_features,
            test_features,
            columns=GENES + CELLS,
            config=config["QuantileTransformer"],
        )
        logging.info("Normalize by RankGauss.")
    elif config["norm_type"] == "zscore":
        train_features, test_features = apply_zscore(
            train_features, test_features, columns=GENES + CELLS
        )
        logging.info("Normalize by zscore.")

    dpgmm = BayesianGaussianMixture(**config["BayesianGaussianMixture_g"])
    dpgmm.fit(train_features[GENES])
    with open(
        os.path.join(args.outdir, f"dpgmm_{config['norm_type']}_g.job"), "wb"
    ) as f:
        joblib.dump(dpgmm, f)
    proba = dpgmm.predict_proba(train_features[GENES])
    plt.figure()
    plt.imshow(proba, aspect="auto")
    plt.title("train_dpgmm_g")
    plt.colorbar()
    plt.savefig(os.path.join(args.outdir, "train_dpgmm_g.png"))
    plt.close()
    proba = dpgmm.predict_proba(test_features[GENES])
    plt.figure()
    plt.imshow(proba, aspect="auto")
    plt.title("test_dpgmm_g")
    plt.colorbar()
    plt.savefig(os.path.join(args.outdir, "test_dpgmm_g.png"))
    plt.close()
    logging.info("finish g.")
    dpgmm = BayesianGaussianMixture(**config["BayesianGaussianMixture_c"])
    dpgmm.fit(train_features[CELLS])
    with open(
        os.path.join(args.outdir, f"dpgmm_{config['norm_type']}_c.job"), "wb"
    ) as f:
        joblib.dump(dpgmm, f)
    proba = dpgmm.predict_proba(train_features[CELLS])
    plt.figure()
    plt.imshow(proba, aspect="auto")
    plt.title("train_dpgmm_c")
    plt.colorbar()
    plt.savefig(os.path.join(args.outdir, "train_dpgmm_c.png"))
    plt.close()
    proba = dpgmm.predict_proba(test_features[CELLS])
    plt.figure()
    plt.imshow(proba, aspect="auto")
    plt.title("test_dpgmm_c")
    plt.colorbar()
    plt.savefig(os.path.join(args.outdir, "test_dpgmm_c.png"))
    plt.close()
    logging.info("finish c.")


if __name__ == "__main__":
    main()
