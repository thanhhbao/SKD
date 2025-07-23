from typing import Dict, Type, Optional
import torch.nn as nn
import timm
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

def build_model(name, num_classes=2, pretrained=True, **kwargs):
    if name == 'vit_base_patch16_224':
        model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    elif name == 'convnextv2_tiny':
        model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    elif name == 'densenet121':
        model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")
    
    return model