import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import LOSS

@LOSS.register('MSE')
class MSE:
  """
  Mean Squared Error loss.
  """
  def __init__(self, *args, **kwargs):
    pass

  def __call__(self, *args, **kwargs):
    return nn.functional.mse_loss(*args, **kwargs)

@LOSS.register('BinaryCrossEntropy')
class BinaryCrossEntropy:
  """
  Binary Cross Entropy loss.
  """
  def __init__(self, *args, **kwargs):
    pass

  def __call__(self, *args, **kwargs):
    return nn.functional.binary_cross_entropy(*args, **kwargs)

@LOSS.register('BinaryCrossEntropyWithLogits')
class BinaryCrossEntropyWithLogits:
  """
  Binary Cross Entropy with Logits loss.
  """
  def __init__(self, *args, **kwargs):
    pass

  def __call__(self, *args, **kwargs):
    return nn.functional.binary_cross_entropy_with_logits(*args, **kwargs)
  
@LOSS.register('CrossEntropy')
class CrossEntropy:
  """
  Cross Entropy loss.
  """
  def __init__(self, *args, **kwargs):
    pass
  
  def __call__(self, *args, **kwargs):
    return nn.functional.cross_entropy(*args, **kwargs)

@LOSS.register('FocalLoss')
class FocalLoss:
    def __init__(self, alpha=1.0, gamma=4.0, reduction='mean'):
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32, requires_grad=False)
        elif isinstance(alpha, list):
            self.alpha = torch.tensor(alpha, dtype=torch.float32, requires_grad=False)
        else:
            raise TypeError("alpha must be float or list")
        self.gamma = gamma
        self.reduction = reduction

    def __call__(self, inputs, targets):
        if targets.dim() != 1:
            targets = torch.argmax(targets, dim=1)
        targets = targets.long()

        if self.alpha.device != inputs.device:
            self.alpha = self.alpha.to(inputs.device)

        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        if self.alpha.dim() > 0:  # alpha for each class
            alpha_t = self.alpha[targets]
        else:
            alpha_t = self.alpha

        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss