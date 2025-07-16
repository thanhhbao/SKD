import torch
import torch.nn as nn
import timm
from typing import Any
from .registry import BACKBONE

class ConvNeXt(nn.Module):
  """
    ConvNeXt model template using timm.
  """
  def __init__(self, model_name: str, num_labels: int):
    super().__init__()
    self.backbone = timm.create_model(model_name)
    in_feats = self.backbone.head.fc.in_features
    self.backbone.head.fc = nn.Linear(in_feats, num_labels)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    logits = self.backbone(x)
    return logits #nn.functional.softmax(logits, dim=1) 

@BACKBONE.register()
class ConvNeXtV2Tiny(ConvNeXt):
  """
  ConvNeXtV2Tiny model implementation using timm.
  """
  def __init__(self, num_labels: int, *args: Any, **kwargs: Any):
    super().__init__('convnextv2_tiny', num_labels)

@BACKBONE.register()
class ConvNeXtV2Small(ConvNeXt):
  """
  ConvNeXtV2Small model implementation using timm.
  """
  def __init__(self, num_labels: int, *args: Any, **kwargs: Any):
    super().__init__('convnextv2_small', num_labels)

@BACKBONE.register()
class ConvNeXtV2Pico(ConvNeXt):
  """
  ConvNeXtV2Pico model implementation using timm.
  """
  def __init__(self, num_labels: int, *args: Any, **kwargs: Any):
    super().__init__('convnextv2_pico', num_labels)

@BACKBONE.register()
class ConvNeXtV2Nano(ConvNeXt):
  """
  ConvNeXtV2Nano model implementation using timm.
  """
  def __init__(self, num_labels: int, *args: Any, **kwargs: Any):
    super().__init__('convnextv2_nano', num_labels)

@BACKBONE.register()
class ConvNeXtV2Base(ConvNeXt):
  """
  ConvNeXtV2Base model implementation using timm.
  """
  def __init__(self, num_labels: int, *args: Any, **kwargs: Any):
    super().__init__('convnextv2_base', num_labels)

@BACKBONE.register()
class ConvNeXtV2Huge(ConvNeXt):
  """
  ConvNeXtV2Huge model implementation using timm.
  """
  def __init__(self, num_labels: int, *args: Any, **kwargs: Any):
    super().__init__('convnextv2_huge', num_labels)