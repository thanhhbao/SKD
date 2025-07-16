import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import LOSS

@LOSS.register()
class MSE:
  """
  Mean Squared Error loss.
  """
  def __init__(self, *args, **kwargs):
    pass

  def __call__(self, *args, **kwargs):
    return nn.functional.mse_loss(*args, **kwargs)

@LOSS.register()
class BinaryCrossEntropy:
  """
  Binary Cross Entropy loss.
  """
  def __init__(self, *args, **kwargs):
    pass

  def __call__(self, *args, **kwargs):
    return nn.functional.binary_cross_entropy(*args, **kwargs)

@LOSS.register()
class BinaryCrossEntropyWithLogits:
  """
  Binary Cross Entropy with Logits loss.
  """
  def __init__(self, *args, **kwargs):
    pass

  def __call__(self, *args, **kwargs):
    return nn.functional.binary_cross_entropy_with_logits(*args, **kwargs)
  
@LOSS.register()
class CrossEntropy:
  """
  Cross Entropy loss.
  """
  def __init__(self, *args, **kwargs):
    pass
  
  def __call__(self, *args, **kwargs):
    return nn.functional.cross_entropy(*args, **kwargs)

@LOSS.register()
class FocalLoss:
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', **kwargs):
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor(alpha)
        elif isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)  # shape (num_classes)

    def __call__(self, inputs, targets, **kwargs):
        if targets.dim() != 1:
            targets = torch.argmax(targets, dim=1)
        targets = targets.long()

        ce_loss = F.cross_entropy(inputs, targets, reduction='none', **kwargs)
        pt = torch.exp(-ce_loss)

        if self.alpha.dim() > 0:  # alpha per class
            alpha_t = self.alpha[targets]  # gather alpha for each sample
        else:
            alpha_t = self.alpha

        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
