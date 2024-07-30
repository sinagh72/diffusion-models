import torch
import torch.nn as nn
from models.base import BaseNet
import torch.nn.functional as F

from models.focal_loss import FocalLoss


class MLP(nn.Module):
    def __init__(self, in_features, num_classes, dropout_rate=0.2):
        super(MLP, self).__init__()
        self.dense1 = nn.Linear(in_features=in_features, out_features=in_features * 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dense2 = nn.Linear(in_features=in_features * 2, out_features=in_features)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(in_features=in_features, out_features=num_classes)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.dense1(x)))
        x = self.dropout2(self.relu2(self.dense2(x)))
        x = self.output_layer(x)
        return x


class CLS(BaseNet):
    def __init__(self, feature_dim, **kwargs):
        super().__init__(**kwargs)
        self.encoder = kwargs["encoder"]
        self.cls_head = MLP(feature_dim, len(kwargs["classes"]))
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.criterion = FocalLoss()

    def _calculate_loss(self, batch):
        imgs, labels = batch
        # get the latent features
        z = self.encoder.autoencoder.encode_stage_2_inputs(imgs) * self.encoder.inferer.scale_factor
        timesteps = torch.randint(0, self.encoder.inferer.scheduler.num_train_timesteps, (len(z),)).to(self.device)
        noise = torch.randn_like(z).to(self.device)
        # add t steps of noise to the input image
        noisy_z = self.encoder.inferer.scheduler.add_noise(z, noise, timesteps)
        # extract features
        pred_z = self.encoder.extract_features(noisy_z, timesteps)
        avg_z = self.adaptive_avg_pool(pred_z)
        preds = self.cls_head(avg_z.view(avg_z.size(0), -1))
        loss = self.criterion(preds, labels)
        preds = preds.argmax(dim=-1) if len(self.classes) == 2 else preds
        return {"loss": loss, "preds": preds, "labels": labels}


class Classifier(BaseNet):
    def __init__(self, feature_dim, **kwargs):
        super().__init__(**kwargs)
        if feature_dim != 0:
            self.encoder = nn.Sequential(kwargs["encoder"],
                                         MLP(feature_dim, len(kwargs["classes"]))
                                         )
        else:
            self.encoder = kwargs["encoder"]

        self.criterion = FocalLoss()

    def _calculate_loss(self, batch):
        imgs, labels = batch
        preds = self.forward(imgs)
        # loss = F.cross_entropy(preds, labels, weight=self.class_weights.to(self.device))
        # loss = F.cross_entropy(preds, labels)
        loss = self.criterion(preds, labels)
        preds = preds.argmax(dim=-1) if len(self.classes) == 2 else preds
        return {"loss": loss, "preds": preds, "labels": labels}
