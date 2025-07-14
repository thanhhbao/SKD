import torch
import torchmetrics
# Removed matplotlib.pyplot and seaborn as they are not needed for console logging

class MetricsLogger:
    """
    Metrics logging system for tracking model performance metrics
    across different stages of training and evaluation.
    """

    def __init__(self, config, device='cuda'):
        """
        Initialize metrics logger based on configuration.
        """
        self.device = device
        self.metrics = {}
        self.stages = ['train', 'val', 'test']
        self.config = config # Store config to access fbeta_beta, num_classes etc.

        # Create metrics for each stage based on config
        for stage in self.stages:
            self.metrics[stage] = self._create_metrics_for_stage(config)

        # Store predictions and targets for Confusion Matrix computation later
        self.all_preds = {stage: [] for stage in self.stages} # <-- ĐÃ THÊM LẠI
        self.all_targets = {stage: [] for stage in self.stages} # <-- ĐÃ THÊM LẠI

    def _create_metrics_for_stage(self, config):
        """
        Create metrics for a specific stage based on configuration.
        """
        task = config.get('task', 'binary')
        num_classes = config.get('num_classes', 2) # Default to 2 for binary classification
        average = config.get('average', 'macro')

        metrics_dict = {}

        # Basic metrics that apply to most tasks
        metrics_dict['accuracy'] = torchmetrics.Accuracy(task=task, num_classes=num_classes).to(self.device)

        # Add AUROC for classification tasks
        if task in ['binary', 'multiclass', 'multilabel']:
            metrics_dict['auroc'] = torchmetrics.AUROC(
                task=task,
                num_classes=num_classes,
                average=average
            ).to(self.device)

        # Add F1 score for classification
        if task in ['binary', 'multiclass', 'multilabel']:
            metrics_dict['f1'] = torchmetrics.F1Score(
                task=task,
                num_classes=num_classes,
                average=average
            ).to(self.device)

        # Add more metrics based on configuration
        if config.get('include_precision', False):
            metrics_dict['precision'] = torchmetrics.Precision(
                task=task,
                num_classes=num_classes,
                average=average
            ).to(self.device)

        if config.get('include_recall', False):
            metrics_dict['recall'] = torchmetrics.Recall(
                task=task,
                num_classes=num_classes,
                average=average
            ).to(self.device)

        # --- Thêm FBeta ---
        if config.get('include_fbeta', False): # <-- ĐÃ THÊM LẠI LOGIC NÀY
            fbeta_beta = config.get('fbeta_beta', 1.0)
            metrics_dict[f'fbeta_{fbeta_beta:.1f}'.replace('.', '')] = torchmetrics.FBetaScore(
                task=task,
                num_classes=num_classes,
                average=average,
                beta=fbeta_beta
            ).to(self.device)

        # --- Add Confusion Matrix metric (to be computed and printed) ---
        if config.get('include_confusion_matrix', False): # <-- ĐÃ THÊM LẠI LOGIC NÀY
            metrics_dict['confusion_matrix'] = torchmetrics.ConfusionMatrix(
                task=task,
                num_classes=num_classes
            ).to(self.device)

        # --- Add Score (If it's a custom metric) ---
        # If 'Score' is a specific custom metric, define its logic here.
        # For example, if it's just another name for Accuracy:
        # if config.get('include_score', False):
        #     metrics_dict['score'] = torchmetrics.Accuracy(task=task, num_classes=num_classes).to(self.device)

        return metrics_dict

    def update(self, stage: str, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Update metrics for the given stage with predictions and ground truth.
        """
        if stage not in self.metrics:
            raise ValueError(f"Unknown stage: {stage}")

        # Store predictions and targets for Confusion Matrix
        self.all_preds[stage].append(y_pred.detach().cpu()) 
        self.all_targets[stage].append(y_true.detach().cpu()) 

        # Update each metric
        for _, metric in self.metrics[stage].items():
            metric.update(y_pred, y_true)

    def compute_and_log(self, stage: str, logger_fn, prefix: str=''):
        """
        Compute metrics for the stage and log them using the provided logger function.
        Prints Confusion Matrix directly to console.
        Returns a dictionary of computed scalar metrics.
        """
        if stage not in self.metrics:
            raise ValueError(f"Unknown stage: {stage}")

        computed_scalar_metrics = {}

        for name, metric in self.metrics[stage].items():
            if name == 'confusion_matrix':
                continue # Handle CM separately below

            value = metric.compute()
            metric_name = f"{prefix}{stage}_{name}"
            logger_fn(metric_name, value, prog_bar=True) # Use logger_fn (self.log from ModelWrapper)
            computed_scalar_metrics[metric_name] = value.item() # Store scalar value

        # --- Compute and Print Confusion Matrix directly to console ---
        if 'confusion_matrix' in self.metrics[stage]: 
            # Concatenate all stored tensors for CM
            all_preds_stage = torch.cat(self.all_preds[stage]).to(self.device)
            all_targets_stage = torch.cat(self.all_targets[stage]).to(self.device)

            cm_metric = self.metrics[stage]['confusion_matrix']
            cm_metric.update(all_preds_stage, all_targets_stage)
            confusion_matrix_tensor = cm_metric.compute()

            # Print Confusion Matrix to console
            print(f"\n{prefix}{stage}_confusion_matrix:\n{confusion_matrix_tensor.int().cpu().numpy()}")

            # Reset CM metric after computation
            cm_metric.reset()

        return computed_scalar_metrics # Now only returns scalar metrics

    def reset(self, stage=None):
        """
        Reset metrics for the given stage or all stages.
        """
        if stage is None:
            for s in self.stages:
                self._reset_stage(s)
        elif stage in self.metrics:
            self._reset_stage(stage)
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def _reset_stage(self, stage):
        """
        Reset all metrics for a specific stage.
        """
        for metric in self.metrics[stage].values():
            metric.reset()
        # Reset stored predictions and targets
        self.all_preds[stage] = [] 
        self.all_targets[stage] = [] 
