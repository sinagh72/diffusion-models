import torch
import torch.nn.functional as F
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # This assumes that 'inputs' is the output of a softmax or log_softmax layer,
        # and 'targets' is a batch of ground-truth class indices.
        # If using log_softmax, you should apply torch.exp to 'inputs' to get the probabilities.
        log_prob = F.log_softmax(inputs, dim=-1)
        prob = torch.exp(log_prob)

        # Gather the probabilities of the targeted classes
        targets_prob = prob.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        # Calculate the focal loss
        focal_loss = -self.alpha * ((1 - targets_prob) ** self.gamma) * log_prob.gather(dim=-1, index=targets.unsqueeze(
            -1)).squeeze(-1)

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss
