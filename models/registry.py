from typing import Dict, Type, Optional
import torch.nn as nn
from utils.base_registry import Registry

class ModelRegistry(Registry):
  """
  Registry for model classes.
  """
  def __init__(self):
    super().__init__('ModelRegistry')

  def get_model(self, name: str, *args, **kwargs) -> nn.Module:
    return self.get(name, *args, **kwargs)

  def list_models(self) -> list:
    return self.list_all()

# Create global registry instance
BACKBONE = ModelRegistry() 

def build_model(name, num_labels, **kwargs):
  return BACKBONE.get_model(name, num_labels, **kwargs)
