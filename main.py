import os
import argparse
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from models import ModelWrapper
from utils import DataLoaderWrapper, TrainerWrapper
from utils import load_config, get_default_config_path, save_config


class EpochMetricsPrinter(Callback):
    def __init__(self):
        self.metric_map = [
            ('train_loss', 'train_loss'),
            ('val_loss', 'val_loss'),
            ('train_f1', 'epoch_train_f1'),
            ('val_f1', 'epoch_val_f1'),
            ('val_precision', 'epoch_val_precision'),
            ('val_recall', 'epoch_val_recall'),
            ('val_accuracy', 'epoch_val_accuracy'),
            ('val_auroc', 'epoch_val_auroc'),
        ]
        self.colors = {
            'reset': '\033[0m',
            'bold': '\033[1m',
            'cyan': '\033[96m',
            'green': '\033[92m',
            'warning': '\033[94m',
            'red': '\033[91m',
            'dim': '\033[90m',
        }

    def _to_float(self, value):
        try:
            return float(value.detach().cpu().item())
        except Exception:
            try:
                return float(value)
            except Exception:
                return None

    def _format_confusion_matrix(self, metrics):
        cm_keys = [
            'epoch_val_confmat_00',
            'epoch_val_confmat_01',
            'epoch_val_confmat_10',
            'epoch_val_confmat_11',
        ]
        if not all(key in metrics for key in cm_keys):
            return None

        cm00 = int(self._to_float(metrics['epoch_val_confmat_00']) or 0)
        cm01 = int(self._to_float(metrics['epoch_val_confmat_01']) or 0)
        cm10 = int(self._to_float(metrics['epoch_val_confmat_10']) or 0)
        cm11 = int(self._to_float(metrics['epoch_val_confmat_11']) or 0)

        return [[cm00, cm01], [cm10, cm11]]

    def _colorize(self, text, color):
        return f"{self.colors[color]}{text}{self.colors['reset']}"

    def _metric_color(self, metric_name, value):
        if metric_name in {'train_loss', 'val_loss'}:
            if value <= 0.25:
                return 'green'
            if value <= 0.50:
                return 'warning'
            return 'red'

        if metric_name in {'train_f1', 'val_f1', 'val_precision', 'val_recall', 'val_accuracy', 'val_auroc'}:
            if value >= 0.85:
                return 'green'
            if value >= 0.70:
                return 'warning'
            return 'red'

        return 'cyan'

    def _notes(self, metric_values):
        notes = []
        val_f1 = metric_values.get('val_f1')
        val_recall = metric_values.get('val_recall')
        val_precision = metric_values.get('val_precision')
        val_accuracy = metric_values.get('val_accuracy')
        val_auroc = metric_values.get('val_auroc')

        if val_accuracy is not None and val_f1 is not None and (val_accuracy - val_f1) >= 0.12:
            notes.append(('warning', 'Accuracy cao hon F1 kha nhieu: can coi chung anh huong mat can bang lop.'))

        if val_recall is not None and val_recall < 0.70:
            notes.append(('red', 'Recall con thap: model dang bo sot kha nhieu ca duong tinh.'))
        elif val_recall is not None and val_recall < 0.80:
            notes.append(('warning', 'Recall chua tot: nen uu tien cai thien bat duong tinh.'))

        if val_precision is not None and val_recall is not None and abs(val_precision - val_recall) >= 0.10:
            notes.append(('warning', 'Precision va Recall lech nhau ro: can can nhac dieu chinh threshold.'))

        if val_auroc is not None and val_auroc < 0.75:
            notes.append(('red', 'AUROC thap: kha nang tach lop cua model chua tot.'))

        if val_f1 is not None and val_f1 >= 0.85:
            notes.append(('green', 'F1 tot va can bang hon Accuracy cho bai toan mat can bang.'))

        return notes

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        lines = []
        metric_values = {}
        for display_name, metric_key in self.metric_map:
            value = metrics.get(metric_key)
            if value is None:
                continue
            value = self._to_float(value)
            if value is None:
                continue
            metric_values[display_name] = value
            colored_value = self._colorize(f"{value:.4f}", self._metric_color(display_name, value))
            lines.append(f"{display_name:<18}: {colored_value}")

        confusion_matrix = self._format_confusion_matrix(metrics)

        if lines:
            print("\n" + self._colorize("=" * 60, 'cyan'))
            print(self._colorize(f"Epoch {trainer.current_epoch + 1} Summary", 'bold'))
            print(self._colorize("-" * 60, 'dim'))
            for line in lines:
                print(line)
            if confusion_matrix is not None:
                tn, fp = confusion_matrix[0]
                fn, tp = confusion_matrix[1]
                fp_color = 'warning' if fp > fn else 'green'
                fn_color = 'red' if fn > fp else 'warning'
                print("val_confusion_matrix:")
                print(
                    f"[[{self._colorize(f'{tn:4d}', 'green')}, {self._colorize(f'{fp:4d}', fp_color)}],"
                )
                print(
                    f" [{self._colorize(f'{fn:4d}', fn_color)}, {self._colorize(f'{tp:4d}', 'green')}]]"
                )

            notes = self._notes(metric_values)
            if notes:
                print(self._colorize("-" * 60, 'dim'))
                print(self._colorize("Nhan xet:", 'bold'))
                for color, note in notes:
                    print(self._colorize(f"- {note}", color))

            print(self._colorize("=" * 60, 'cyan'))


def parse_args():
    parser = argparse.ArgumentParser(description='Skin cancer detection pipeline')
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('-e', '--exp_name', type=str, default=None, help='Experiment name (overrides config)')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config_path = args.config if args.config else get_default_config_path()
    config = load_config(config_path)

    # Override experiment name
    config['experiment']['name'] = args.exp_name if args.exp_name else config['experiment']['name']

    # Extract config sections
    exp_config = config['experiment']
    data_config = config['data']
    trainer_config = config['trainer']
    metrics_config = config['metrics']
    metrics_config['num_classes'] = data_config['num_classes']
    model_config = config['model']
    model_config['num_labels'] = data_config['num_classes']
    model_config['_metrics_config'] = metrics_config

    # Output directory
    output_dir = os.path.join(exp_config['base_output_dir'], exp_config['name'])
    os.makedirs(output_dir, exist_ok=True)
    save_config(config, os.path.join(output_dir, 'config.yaml'))

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='{epoch}-{val_loss:.4f}',
        monitor='val_loss',
        save_top_k=exp_config['save_top_k'],
        mode='min',
        every_n_epochs=exp_config['check_val_every_n_epoch'],
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min',
        min_delta=1e-4,
        verbose=True,
    )

    epoch_metrics_printer = EpochMetricsPrinter()

    # Loggers
    csv_logger = CSVLogger(save_dir=output_dir, name='metrics_logs')
    tensorboard_logger = TensorBoardLogger(save_dir=output_dir, name='tensorboard_logs')

    # Trainer config
    trainer_config.update({
        'check_val_every_n_epoch': exp_config['check_val_every_n_epoch'],
        'callbacks': [checkpoint_callback, early_stopping_callback, epoch_metrics_printer],
        'logger': [csv_logger, tensorboard_logger],
        'enable_progress_bar': False,
        'enable_model_summary': False,
    })

    # Data & model
    data_wrapper = DataLoaderWrapper(**data_config)
    data_wrapper.setup()

    model_wrapper = ModelWrapper(**model_config)
    checkpoint_path = config.get('resume', {}).get('from_checkpoint', None)
    trainer = TrainerWrapper(checkpoint=checkpoint_path, config=config, **trainer_config)

    # Train
    trainer.fit(model_wrapper, data_wrapper.train_dataloader(), data_wrapper.val_dataloader())

    # Evaluate with thresholds
    print("\nEvaluating model with multiple thresholds...")
    best_th, all_results = trainer.evaluate_with_thresholds(model_wrapper, data_wrapper.val_dataloader())
    best_metrics = all_results[best_th]
    priority_metric = config['metrics'].get('metric_priority', 'f1')

    print("\nBest threshold summary")
    print(f"threshold            : {best_th:.2f}")
    print(f"selected_by          : {priority_metric}")
    print(f"accuracy             : {best_metrics['accuracy']:.4f}")
    print(f"precision            : {best_metrics['precision']:.4f}")
    print(f"recall               : {best_metrics['recall']:.4f}")
    print(f"f1_score             : {best_metrics['f1']:.4f}")
    print(f"confusion_matrix     : {best_metrics['confusion_matrix']}")

    # Save best threshold
    with open(os.path.join(output_dir, 'best_threshold.txt'), 'w') as f:
        f.write(f"Best threshold: {best_th} (based on {priority_metric}: "
                f"{best_metrics[priority_metric]:.4f})")


if __name__ == "__main__":
    main()
