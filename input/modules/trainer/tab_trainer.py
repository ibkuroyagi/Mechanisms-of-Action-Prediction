import logging
import os

from tqdm import tqdm
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from tensorboardX import SummaryWriter


def mean_log_loss(y_true, y_pred, out_dim=206):
    metrics = []
    for i in range(out_dim):
        metrics.append(
            log_loss(y_true[:, i], y_pred[:, i].astype(float), labels=[0, 1])
        )
    return np.mean(metrics)


class TabTrainer(object):
    """Customized trainer module for Mechanisms-of-Action training."""

    def __init__(
        self,
        steps,
        epochs,
        data_loader,
        model,
        criterion,
        optimizer,
        scheduler,
        config,
        device=torch.device("cpu"),
        train=False,
        add_name="",
        is_kernel=False,
    ):
        """Initialize trainer.

        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            data_loader (dict): Dict of data loaders. It must contrain "train" and "dev" loaders.
            model (dict): Dict of models. It must contrain "generator" and "discriminator" models.
            criterion (torch.nn): It must contrain "stft" and "mse" criterions.
            optimizer (dict): Dict of optimizers. It must contrain "generator" and "discriminator" optimizers.
            scheduler (dict): Dict of schedulers. It must contrain "generator" and "discriminator" schedulers.
            config (dict): Config dict loaded from yaml format configuration file.
            device (torch.deive): Pytorch device instance.

        """
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.add_name = add_name
        self.is_kernel = is_kernel
        if train:
            self.writer = SummaryWriter(config["outdir"])
        self.finish_train = False
        self.epoch_train_loss = defaultdict(float)
        self.epoch_eval_loss = defaultdict(float)
        self.train_pred_epoch = np.empty((0, config["out_dim"]))
        self.train_y_epoch = np.empty((0, config["out_dim"]))
        self.dev_pred_epoch = np.empty((0, config["out_dim"]))
        self.dev_y_epoch = np.empty((0, config["out_dim"]))
        self.forward_count = 0
        self.best_loss = 1e9

    def run(self):
        """Run training."""
        self.tqdm = tqdm(
            initial=self.steps, total=self.config["train_max_steps"], desc="[train]"
        )
        while True:
            # train one epoch
            self._train_epoch()

            # check whether training is finished
            if self.finish_train:
                break

        self.tqdm.close()
        if self.is_kernel:
            print("Finished training.")
        else:
            logging.info("Finished training.")

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be saved.

        """
        state_dict = {
            "steps": self.steps,
            "epochs": self.epochs,
        }
        state_dict["model"] = self.model.state_dict()

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.

        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(state_dict["model"])
        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]

    def _train_step(self, batch):
        """Train model one step."""
        # parse batch
        x, y = batch["X"], batch["y"]
        batch_size = x.size(0)
        x = x.to(self.device)
        y = y.to(self.device)  # (B, 206)
        y_ = self.model(x)  # (B, 206)
        if self.config["loss_type"] == "BCELoss":
            loss = self.criterion(torch.sigmoid(y_), y)
        else:
            loss = self.criterion(y_, y)
        loss = loss / self.config["accum_grads"]
        self.epoch_train_loss["train/loss"] += (
            loss.item()
            / (batch_size * self.config["out_dim"] * len(self.data_loader["train"]))
            * self.config["accum_grads"]
        )
        loss.backward()
        self.forward_count += 1
        if self.forward_count == self.config["accum_grads"]:
            # update parameters
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.forward_count = 0

            # update counts
            self.steps += 1
            self.tqdm.update(1)
            self._check_train_finish()
        y = y.detach().cpu().numpy()
        self.train_y_epoch = np.concatenate([self.train_y_epoch, y], axis=0)
        y_ = torch.sigmoid(y_).detach().cpu().numpy()
        self.train_pred_epoch = np.concatenate([self.train_pred_epoch, y_], axis=0)

    def _train_epoch(self):
        """Train model one epoch."""
        self.epoch_train_loss["train/loss"] = 0
        for train_steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            # train one step
            self._train_step(batch)
            self._check_save_interval()
            # check whether training is finished
            if self.finish_train:
                return
        self.epoch_train_loss["train/epoch_metric"] = mean_log_loss(
            y_true=self.train_y_epoch,
            y_pred=self.train_pred_epoch,
        )
        self.epoch_train_loss["train/epoch_auc"] = roc_auc_score(
            y_true=self.train_y_epoch.flatten(), y_score=self.train_pred_epoch.flatten()
        )
        preds = (self.train_pred_epoch > 0.5).flatten()
        self.epoch_train_loss["train/epoch_acc"] = accuracy_score(
            self.train_y_epoch.flatten(), preds
        )
        self.epoch_train_loss["train/epoch_recall"] = recall_score(
            self.train_y_epoch.flatten(), preds
        )
        self.epoch_train_loss["train/epoch_precision"] = precision_score(
            self.train_y_epoch.flatten(), preds, zero_division=0
        )
        self._eval_epoch()
        # log
        if self.is_kerrnel:
            print(
                f"(Steps: {self.steps}) Finished {self.epochs} epoch training "
                f"({train_steps_per_epoch} steps per epoch)."
            )
        else:
            logging.info(
                f"(Steps: {self.steps}) Finished {self.epochs} epoch training "
                f"({train_steps_per_epoch} steps per epoch)."
            )
        for key in self.epoch_train_loss.keys():
            if self.is_kerrnel:
                print(
                    f"(Epoch: {self.epochs}) {key} = {self.epoch_train_loss[key]:.5f}."
                )
            else:
                logging.info(
                    f"(Epoch: {self.epochs}) {key} = {self.epoch_train_loss[key]:.5f}."
                )
        self._write_to_tensorboard(self.epoch_train_loss)
        # update
        self.train_steps_per_epoch = train_steps_per_epoch
        self.epochs += 1
        # reset
        self.train_y_epoch = np.empty((0, self.config["out_dim"]))
        self.train_pred_epoch = np.empty((0, self.config["out_dim"]))
        self.epoch_train_loss = defaultdict(float)

    @torch.no_grad()
    def _eval_step(self, batch):
        """Evaluate model one step."""
        # parse batch
        x, y = batch["X"], batch["y"]
        batch_size = x.size(0)
        x = x.to(self.device)
        y = y.to(self.device)
        y_ = self.model(x)
        # add to total eval loss
        if self.config["loss_type"] == "BCELoss":
            loss = self.criterion(torch.sigmoid(y_), y)
        else:
            loss = self.criterion(y_, y)
        loss = loss / self.config["accum_grads"]
        self.epoch_eval_loss["dev/loss"] += (
            loss.item()
            / (batch_size * self.config["out_dim"] * len(self.data_loader["dev"]))
            * self.config["accum_grads"]
        )
        y_ = torch.sigmoid(y_).cpu().numpy()
        self.dev_pred_epoch = np.concatenate([self.dev_pred_epoch, y_], axis=0)
        y = y.cpu().numpy()
        self.dev_y_epoch = np.concatenate([self.dev_y_epoch, y], axis=0)

    def _eval_epoch(self):
        """Evaluate model one epoch."""
        self.epoch_eval_loss["dev/loss"] = 0
        if self.is_kerrnel:
            print(f"(Steps: {self.steps}) Start dev data's evaluation.")
        else:
            logging.info(f"(Steps: {self.steps}) Start dev data's evaluation.")
        # change mode
        self.model.eval()

        # calculate loss for each batch
        for eval_steps_per_epoch, batch in enumerate(
            tqdm(self.data_loader["dev"], desc="[dev]"), 1
        ):
            # eval one step
            self._eval_step(batch)
        self.epoch_eval_loss["dev/epoch_metric"] = mean_log_loss(
            y_true=self.dev_y_epoch,
            y_pred=self.dev_pred_epoch,
        )
        self.epoch_eval_loss["dev/epoch_auc"] = roc_auc_score(
            y_true=self.dev_y_epoch.flatten(), y_score=self.dev_pred_epoch.flatten()
        )
        preds = (self.dev_pred_epoch > 0.5).flatten()
        self.epoch_eval_loss["dev/epoch_acc"] = accuracy_score(
            self.dev_y_epoch.flatten(), preds
        )
        self.epoch_eval_loss["dev/epoch_recall"] = recall_score(
            self.dev_y_epoch.flatten(), preds
        )
        self.epoch_eval_loss["dev/epoch_precision"] = precision_score(
            self.dev_y_epoch.flatten(), preds, zero_division=0
        )
        if self.best_loss > self.epoch_eval_loss["dev/epoch_metric"]:
            self.save_checkpoint(
                os.path.join(
                    self.config["outdir"],
                    "best",
                    f"best_loss{self.add_name}.pkl",
                )
            )
            self.best_loss = self.epoch_eval_loss["dev/epoch_metric"]
        # log
        if self.is_kerrnel:
            print(
                f"(Steps: {self.steps}) Finished dev data's evaluation "
                f"({eval_steps_per_epoch} steps per epoch)."
            )
        else:
            logging.info(
                f"(Steps: {self.steps}) Finished dev data's evaluation "
                f"({eval_steps_per_epoch} steps per epoch)."
            )
        for key in self.epoch_eval_loss.keys():
            if self.is_kerrnel:
                print(
                    f"(Epoch: {self.epochs}) {key} = {self.epoch_eval_loss[key]:.5f}."
                )
            else:
                logging.info(
                    f"(Epoch: {self.epochs}) {key} = {self.epoch_eval_loss[key]:.5f}."
                )
        # average loss
        if self.is_kerrnel:
            print(f"(Steps: {self.steps}) Start eval data's evaluation.")
        else:
            logging.info(f"(Steps: {self.steps}) Start eval data's evaluation.")
        # update scheduler step
        if self.scheduler is not None:
            self.scheduler.step(self.epoch_eval_loss["dev/epoch_metric"])
        # record
        self._write_to_tensorboard(self.epoch_eval_loss)

        # reset
        self.epoch_eval_loss = defaultdict(float)

        self.dev_pred_epoch = np.empty((0, self.config["out_dim"]))
        self.dev_y_epoch = np.empty((0, self.config["out_dim"]))

        # restore mode
        self.model.train()

    def inference(self):
        """evaluate and save intermediate result."""
        y_preds = torch.empty((0, self.config["out_dim"])).to(self.device)
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.data_loader["eval"]):
                y_batch_ = self.model(batch["X"].to(self.device))
                y_preds = torch.cat([y_preds, y_batch_], dim=0)
            y_preds = torch.sigmoid(y_preds)
        return y_preds.detach().cpu().numpy()

    def _write_to_tensorboard(self, loss):
        """Write to tensorboard."""
        for key, value in loss.items():
            self.writer.add_scalar(key, value, self.steps)

    def _check_save_interval(self):
        if self.steps % self.config["save_interval_steps"] == 0:
            self.save_checkpoint(
                os.path.join(
                    self.config["outdir"],
                    f"{self.steps}",
                    f"checkpoint-{self.steps}steps{self.add_name}.pkl",
                )
            )
            logging.info(f"Successfully saved checkpoint @ {self.steps} steps.")

    def _check_train_finish(self):
        if self.steps >= self.config["train_max_steps"]:
            self.finish_train = True
