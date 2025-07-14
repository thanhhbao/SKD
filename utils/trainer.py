import torch
import torch.nn as nn
import lightning as L
from typing import Iterable
from lightning.pytorch import seed_everything

class TrainerWrapper:
  """
  A wrapper around PyTorch Lightning's Trainer for consistent interface and setup.
  """
  def __init__(self, seed: int=42, checkpoint: str=None, **kwargs):
    """
    Initialize the trainer wrapper.
    """
    self.trainer = L.Trainer(**kwargs)
    self.checkpoint = checkpoint
    seed_everything(seed, workers=True)

  def fit(self, model: nn.Module, train_dataloader: Iterable, val_dataloader: Iterable):
    """
    Train the model.
    """
    self.trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=self.checkpoint) 