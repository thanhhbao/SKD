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
  """
  Focal Loss for handling class imbalance in classification tasks.
  """
  def __init__(self, alpha=None, gamma=2.0, reduction='mean', **kwargs):
    self.alpha = alpha
    if alpha is not None:
      self.alpha = torch.tensor(alpha)
    self.gamma = gamma
    self.reduction = reduction

  def __call__(self, inputs, targets, **kwargs):
    ce_loss = F.cross_entropy(inputs, targets, reduction='none', **kwargs)
    pt = torch.exp(-ce_loss)
    if self.alpha is not None:
      if self.alpha.device != inputs.device:
        self.alpha = self.alpha.to(inputs.device)
      alpha_t = self.alpha.gather(0, targets.data.view(-1))
    else:
      alpha_t = torch.tensor(1.0, device=inputs.device)
    focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
    if self.reduction == 'mean':
      return focal_loss.mean()
    elif self.reduction == 'sum':
      return focal_loss.sum()
    else:
      return focal_loss