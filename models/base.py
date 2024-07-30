import lightning.pytorch as pl
import torch
from torch import optim
import torch.nn.functional as F
from torchmetrics import F1Score, AUROC, PrecisionRecallCurve
from torchmetrics.classification import MulticlassPrecision
from torchmetrics.classification.accuracy import MulticlassAccuracy
from sklearn.metrics import precision_recall_curve, auc


class BaseNet(pl.LightningModule):
    def __init__(self, classes, lr, encoder, wd=0, momentum=0, dampening=0, optimizer="AdamW", beta1=0.9, beta2=0.999,
                 step_size=1000, factor=0.5):
        """

        :param classes (tuple(str, int)): list of tuples, each tuple consists of class name and class index
        :param lr (float): learning rate
        :param weight_decay (float): weight decay of optimizer
        """
        super().__init__()
        task = "binary" if len(classes) == 2 else "multiclass"
        self.metrics_list = ["accuracy", "precision", "f1", "auc", "pr"]
        self.sessions = ["train", "val", "test"]

        self.train_ac = MulticlassAccuracy(num_classes=len(classes), average=None)
        self.val_ac = MulticlassAccuracy(num_classes=len(classes), average=None)
        self.test_ac = MulticlassAccuracy(num_classes=len(classes), average=None)

        self.train_p = MulticlassPrecision(num_classes=len(classes), average=None)
        self.val_p = MulticlassPrecision(num_classes=len(classes), average=None)
        self.test_p = MulticlassPrecision(num_classes=len(classes), average=None)

        self.train_f1 = F1Score(task=task, num_classes=len(classes), average=None)
        self.val_f1 = F1Score(task=task, num_classes=len(classes), average=None)
        self.test_f1 = F1Score(task=task, num_classes=len(classes), average=None)

        self.train_auc = AUROC(task=task, num_classes=len(classes), average=None)
        self.val_auc = AUROC(task=task, num_classes=len(classes), average=None)
        self.test_auc = AUROC(task=task, num_classes=len(classes), average=None)

        self.train_pr = PrecisionRecallCurve(task=task, num_classes=len(classes))
        self.val_pr = PrecisionRecallCurve(task=task, num_classes=len(classes))
        self.test_pr = PrecisionRecallCurve(task=task, num_classes=len(classes))

        self.metrics = {"train": [self.train_ac, self.train_p, self.train_f1, self.train_auc, self.train_pr],
                        "val": [self.val_ac, self.val_p, self.val_f1, self.val_auc, self.val_pr],
                        "test": [self.test_ac, self.test_p, self.test_f1, self.test_auc, self.test_pr],
                        }

        self.step_output = {"train": [], "val": [], "test": []}
        self.mean_log_keys = ["loss"]
        self.lr = lr
        self.wd = wd
        self.momentum = momentum
        self.dampening = dampening
        self.optimizer = optimizer
        self.beta1 = beta1
        self.beta2 = beta2
        self.step_size = step_size
        self.factor = factor
        self.classes = classes
        self.encoder = encoder
        self.test_classes = classes

    def configure_optimizers(self):
        optimizer = None
        if self.optimizer == "AdamW":
            optimizer = optim.AdamW(self.encoder.parameters(), lr=self.lr, weight_decay=self.wd,
                                    betas=(self.beta1, self.beta2))
        elif self.optimizer == "Adam":
            optimizer = optim.Adam(self.encoder.parameters(), lr=self.lr, weight_decay=self.wd)
        elif self.optimizer == "SGD":
            optimizer = optim.SGD(self.encoder.parameters(), lr=self.lr, weight_decay=self.wd,
                                  dampening=self.dampening, momentum=self.momentum)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=self.factor)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

    def forward(self, x):
        return self.encoder(x)

    def _calculate_loss(self, batch):
        imgs, labels = batch
        preds = self.forward(imgs)
        loss = F.cross_entropy(preds, labels)
        preds = preds.argmax(dim=-1) if len(self.classes) == 2 else preds
        return {"loss": loss, "preds": preds, "labels": labels}

    def training_step(self, batch, batch_idx):
        output = self._calculate_loss(batch)
        self.step_output["train"].append(output)
        return output

    def on_train_epoch_end(self):
        self.stack_update(session="train")

    def validation_step(self, batch, batch_idx):
        output = self._calculate_loss(batch)
        self.step_output["val"].append(output)
        return output

    def on_validation_epoch_end(self):
        self.stack_update(session="val")

    def test_step(self, batch, batch_idx):
        output = self._calculate_loss(batch)
        self.step_output["test"].append(output)
        return output

    def on_test_epoch_end(self, ):
        self.stack_update(session="test")

    def update_metrics(self, session, preds, labels):
        for metric in self.metrics[session]:
            metric.update(preds, labels)

    def stack_update(self, session):
        all_preds = torch.cat([out["preds"] for out in self.step_output[session]])
        all_labels = torch.cat([out["labels"] for out in self.step_output[session]])
        log = {}
        for key in self.mean_log_keys:
            log[f"{session}_{key}"] = torch.stack([out[key] for out in self.step_output[session]]).mean()

        self.update_metrics(session=session, preds=all_preds, labels=all_labels)
        res = self.compute_metrics(session=session)
        self.add_log(session, res, log)
        if session == "test":
            _, self.preds = torch.max(all_preds, 1)
            self.labels = all_labels
        self.log_dict(log, sync_dist=True, on_epoch=True, prog_bar=True, logger=True)
        self.restart_metrics(session=session)

        return all_preds, all_labels

    def compute_metrics(self, session):
        res = {}
        for metric, metric_name in zip(self.metrics[session], self.metrics_list):
            res[metric_name] = metric.compute()
        return res

    def restart_metrics(self, session):
        for metric in self.metrics[session]:
            metric.reset()
        self.step_output[session].clear()  # free memory

    def add_log(self, session, res, log):
        for metric in self.metrics_list:
            log[session + "_" + metric] = 0
        if "pr" in self.metrics_list:
            precision, recall, _ = res["pr"]
            indices = torch.tensor(list(sorted(self.test_classes.values())))
            # Calculating area under the curve for each class
            aucs = []
            for i in indices:
                # Sort the recall and precision arrays
                recall_sorted_indices = recall[i].argsort()
                sorted_recall = recall[i][recall_sorted_indices]
                sorted_precision = precision[i][recall_sorted_indices]

                # Calculate AUC using sklearn's auc function
                class_auc = auc(sorted_recall.cpu().numpy(), sorted_precision.cpu().numpy())
                aucs.append(class_auc)

            # Average AUC across all classes
            average_auc = sum(aucs) / len(aucs)
            log[session + "_pr"] = average_auc
            # log[session + "_pr"] = res["pr"]
        for idx, key in enumerate(self.test_classes.keys()):
            log[session + "_precision_" + key] = res["precision"][idx]
            log[session + "_precision"] += res["precision"][idx]
            log[session + "_accuracy_" + key] = res["accuracy"][idx]
            log[session + "_accuracy"] += res["accuracy"][idx]
            log[session + "_auc_" + key] = res["auc"][idx]
            log[session + "_auc"] += res["auc"][idx]
            log[session + "_f1_" + key] = res["f1"][idx]
            log[session + "_f1"] += res["f1"][idx]
        log[session + "_f1"] /= len(self.test_classes)
        log[session + "_accuracy"] /= len(self.test_classes)
        log[session + "_precision"] /= len(self.test_classes)
        log[session + "_auc"] /= len(self.test_classes)
