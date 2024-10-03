import os
from copy import deepcopy
import lightning.pytorch as pl
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

from utils.log_results import log_results


# def plot_confusion_matrix():
#     plt.rcParams.update({'font.family': 'Times New Roman'})
#     conf_matrix = confusion_matrix(model.labels.cpu().numpy(),
#                                    model.preds.cpu().numpy())
#     true_label = deepcopy(model.test_classes)
#     if client_name == "DS1" and c_name == "DS4":
#         true_label["DME"] = 2
#     plt.figure(figsize=(10, 7))
#     sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues',
#                 xticklabels=dict(sorted(model.classes.items(), key=lambda item: item[1])),
#                 yticklabels=dict(sorted(true_label.items(), key=lambda item: item[1])))
#     plt.xlabel('Predicted labels')
#     plt.ylabel('True labels')
#     plt.title('Confusion Matrix')
#     suffix = "./confusion matrix/" + client_name
#     if not os.path.exists(suffix):
#         os.makedirs(suffix)
#     plt.savefig(suffix + "/" + c_name + "_" + comments)


def evaluate_model(model, classes, dataset_name, data_modules, devices, comments, epochs=100):
    for data_module in data_modules:
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=devices,
            max_epochs=epochs,
        )
        test_loader = data_module.filtered_test_dataloader(classes)
        model.test_classes = data_module.filtered_classes
        if len(model.test_classes) == 0:  #it is not possible to test it!
            continue
        test_results = trainer.test(model, dataloaders=test_loader, verbose=False)
        # Ensure test_preds and test_labels are on CPU and converted to numpy arrays
        # ###
        log_results(classes=data_module.filtered_classes,
                    results=test_results,
                    comments=dataset_name + "_" + comments,
                    test_name=data_module.dataset_name,
                    approach="centralized",
                    epochs=epochs)
