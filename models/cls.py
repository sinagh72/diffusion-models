import torch
import torch.nn as nn
from models.base import BaseNet
import torch.nn.functional as F

from models.focal_loss import FocalLoss


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, 1)  # Bidirectional, so hidden_size * 2
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_output):
        # lstm_output: (batch_size, seq_len, hidden_size * 2)
        attn_weights = self.attn(lstm_output)  # Shape: (batch_size, seq_len, 1)
        attn_weights = self.softmax(attn_weights)  # Shape: (batch_size, seq_len, 1)

        # Multiply attention weights with LSTM output and sum to get the context vector
        context = torch.sum(attn_weights * lstm_output, dim=1)  # Shape: (batch_size, hidden_size * 2)

        return context, attn_weights

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

class LSTMClassifier(BaseNet):
    def __init__(self, feature_dim, **kwargs):
        super().__init__(**kwargs)

        self.encoder = kwargs["encoder"]
        self.lstm = nn.LSTM(input_size=1000, hidden_size=feature_dim, num_layers=2, dropout=0.1,
                            batch_first=True, bidirectional=True)
        # Attention mechanism
        self.attention = Attention(feature_dim)
        # Fully connected layer for classification
        self.fc = nn.Linear(feature_dim*2, len(kwargs["classes"]))
        self.criterion = FocalLoss()



    def _calculate_loss(self, batch):
        img_t, img_t_plus, labels = batch
        cnn_features = []
        out1 = self.forward(img_t)
        out2 = self.forward(img_t_plus)

        cnn_features.append(out1)
        cnn_features.append(out2)

        cnn_features_seq = torch.stack(cnn_features, dim=1)
        # Pass the sequence of CNN features to LSTM
        lstm_out, (hn, cn) = self.lstm(cnn_features_seq)
        # Take the output of the last time step
        # final_output = lstm_out[:, -1, :]
        # Pass through fully connected layer
        # preds = self.fc(final_output)
        # Apply attention to the LSTM output
        context, attn_weights = self.attention(lstm_out)  # context shape: (batch_size, hidden_size * 2)
        preds = self.fc(context)

        # loss = F.cross_entropy(preds, labels, weight=self.class_weights.to(self.device))
        # loss = F.cross_entropy(preds, labels)
        loss = self.criterion(preds, labels)
        preds = preds.argmax(dim=-1) if len(self.classes) == 2 else preds
        return {"loss": loss, "preds": preds, "labels": labels}

class LSTMAttentionClassifier(BaseNet):
    def __init__(self, feature_dim, **kwargs):
        super().__init__(**kwargs)

        self.diffusion = kwargs["encoder"]
        self.lstm = nn.LSTM(input_size=4096, hidden_size=feature_dim, num_layers=2, dropout=0.1,
                            batch_first=True, bidirectional=True)
        # Define a 2D Adaptive Average Pooling layer to reduce spatial dimensions to (4, 4)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # Reduces (32, 32) to (4, 4)
        # Attention mechanism
        self.attention = Attention(feature_dim)
        # Fully connected layer for classification
        self.fc = nn.Linear(feature_dim * 2, len(kwargs["classes"]))
        self.criterion = FocalLoss()
        self.scheduler.set_timesteps(num_inference_steps=90)

    def _calculate_loss(self, batch):
        img_t, img_t_plus, labels = batch
        cnn_features = []
        noise_t, timesteps_t = self.diffusion.noise_timestep(img_t)
        noisy_img_t = self.diffusion.scheduler.add_noise(img_t, noise_t, timesteps_t)
        latent_t = self.diffusion.semantic_encoder(img_t)
        out1 = self.diffusion.unet.encode(noisy_img_t, timesteps_t, context=latent_t.unsqueeze(2))
        out1 = self.pool(out1)

        noise_t_plus, timesteps_t_plus = self.diffusion.noise_timestep(img_t_plus)
        noisy_img_t_plus = self.diffusion.scheduler.add_noise(img_t_plus, noise_t_plus, timesteps_t_plus)
        latent_t_plus = self.diffusion.semantic_encoder(img_t_plus)
        out2 = self.diffusion.unet.encode(noisy_img_t_plus, timesteps_t_plus, context=latent_t_plus.unsqueeze(2))
        out2 = self.pool(out2)

        # out1 = self.diffusion.unet.encode(img_t)
        # out2 = self.diffusion.unet.encode(img_t_plus)

        cnn_features.append(out1)
        cnn_features.append(out2)

        cnn_features_seq = torch.stack(cnn_features, dim=1).view(out2.shape[0], 2, -1)
        # Pass the sequence of CNN features to LSTM
        lstm_out, (hn, cn) = self.lstm(cnn_features_seq)
        # Apply attention to the LSTM output
        context, attn_weights = self.attention(lstm_out)  # context shape: (batch_size, hidden_size * 2)
        preds = self.fc(context)

        # loss = F.cross_entropy(preds, labels, weight=self.class_weights.to(self.device))
        # loss = F.cross_entropy(preds, labels)
        loss = self.criterion(preds, labels)
        preds = preds.argmax(dim=-1) if len(self.classes) == 2 else preds
        return {"loss": loss, "preds": preds, "labels": labels}