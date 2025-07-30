from typing import Dict, Type, Optional
import torch.nn as nn
import timm
from utils.base_registry import Registry

class ModelRegistry(Registry):
    def __init__(self):
        super().__init__('ModelRegistry')

    def get_model(self, name: str, *args, **kwargs) -> nn.Module:
        return self.get(name, *args, **kwargs)

    def list_models(self) -> list:
        return self.list_all()

# Global registry instance
BACKBONE = ModelRegistry()

def build_model(name, num_classes=2, pretrained=True, **kwargs):
    model = timm.create_model(name, pretrained=pretrained, num_classes=0)  # no head
    in_features = model.get_classifier().in_features
    model.reset_classifier(num_classes=num_classes)  # add custom head
    return model