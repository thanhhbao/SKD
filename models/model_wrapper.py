import torch
import torch.nn as nn
import lightning as L

from .registry import build_model
from loss.registry import build_loss
from utils.metrics import MetricsLogger

class ModelWrapper(L.LightningModule):
  """
  Lightning module wrapper for model training with metrics logging.
  """
  def __init__(self, name: str="", num_labels: int=1000, lr: float=1e-4, device: str='cuda', **kwargs):
    super().__init__()
    self.lr = lr
    self.weight_decay = kwargs.pop('weight_decay', 0.05)  # Pop để loại bỏ khỏi kwargs

    # Init model
    self.num_labels = num_labels
    self.model = build_model(name, self.num_labels, **kwargs)

    # Init loss: pop loss_name và loss_kwargs
    loss_name = kwargs.pop('loss_name', 'FocalLoss')  # Pop loss_name, mặc định nếu không có
    loss_kwargs = kwargs.pop('loss_kwargs', {})  # Pop loss_kwargs
    self.loss = build_loss(loss_name, **loss_kwargs)  # Truyền loss_name riêng và **loss_kwargs

    # Init metrics
    metrics_config = kwargs.get('_metrics_config', {})
    self.metrics_logger = MetricsLogger(metrics_config, device=device)
    
    # Save hyperparameters for experiment tracking
    self.save_hyperparameters()

  def configure_optimizers(self):
    return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

  def training_step(self, batch, batch_idx):
    x, y = batch['pixel_values'], batch['label'].long()  # long cho FocalLoss (label 0/1)
    y_pred = self.model(x)  # y_pred (batch,2) cho num_classes=2
    loss = self.loss(y_pred, y)
    
    # Log loss
    self.log("train_loss", loss, prog_bar=True)
    
    # Update metrics: dùng y_pred[:,1] (logits lớp positive) cho binary metrics
    self.metrics_logger.update('train', y_pred[:,1], y.float())  # y.float() nếu metrics cần float targets
    
    return {'loss': loss}

  def validation_step(self, batch, batch_idx):
    x, y = batch['pixel_values'], batch['label'].long()
    y_pred = self.model(x)

    loss = self.loss(y_pred, y)
    
    # Log loss
    self.log("val_loss", loss, prog_bar=True)
    
    # Update metrics
    self.metrics_logger.update('val', y_pred[:,1], y.float())
    
    return {'loss': loss}
  
  def test_step(self, batch, batch_idx):
    x, y = batch['pixel_values'], batch['label'].long()
    y_pred = self.model(x)
    loss = self.loss(y_pred, y)
    
    # Log loss
    self.log("test_loss", loss, prog_bar=True)
    
    # Update metrics
    self.metrics_logger.update('test', y_pred[:,1], y.float())
    
    return {'loss': loss}

  def on_train_epoch_end(self):
    # Compute and log metrics
    self.metrics_logger.compute_and_log('train', self.log, prefix='epoch_')
    # Reset metrics
    self.metrics_logger.reset('train')
      
  def on_validation_epoch_end(self):
    # Compute and log metrics
    self.metrics_logger.compute_and_log('val', self.log, prefix='epoch_')
    # Reset metrics
    self.metrics_logger.reset('val')
      
  def on_test_epoch_end(self):
    # Compute and log metrics
    self.metrics_logger.compute_and_log('test', self.log, prefix='epoch_')
    # Reset metrics
    self.metrics_logger.reset('test') 