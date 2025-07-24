import os
import argparse
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from models import ModelWrapper  # Giả sử models/model_wrapper.py
from utils import DataLoaderWrapper, TrainerWrapper  # Từ utils/data_loader.py và utils/trainer.py
from utils import load_config, get_default_config_path, save_config

def parse_args():
    parser = argparse.ArgumentParser(description='Skin cancer detection pipeline')
    parser.add_argument('-c', '--config', type=str, default=None, help='Path to the configuration file', required=True)
    parser.add_argument('-e', '--exp_name', type=str, default=None, help='Experiment name (overrides config)')
    return parser.parse_args()

def main():
    """
    Main function to run the training pipeline.
    """
    args = parse_args()

    # Load config
    config_path = args.config if args.config else get_default_config_path()
    config = load_config(config_path)

    # Override configuration with command line args
    config['experiment']['name'] = args.exp_name if args.exp_name else config['experiment']['name']
    
    # Extract config sections
    exp_config = config['experiment']
    data_config = config['data']
    trainer_config = config['trainer']
    metrics_config = config['metrics']
    metrics_config['num_classes'] = data_config['num_classes']
    model_config = config['model']
    model_config['num_labels'] = data_config['num_classes']
    # Update model config with metrics config
    model_config['_metrics_config'] = metrics_config

    # Create output experiment directory
    output_dir = os.path.join(exp_config['base_output_dir'], exp_config['name'])
    os.makedirs(output_dir, exist_ok=True)

    # Save config for reproducibility
    save_config(config, os.path.join(output_dir, 'config.yaml'))

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        save_top_k=exp_config['save_top_k'],  # Keep only the N most recent checkpoints
        every_n_epochs=exp_config['check_val_every_n_epoch'],  # Save checkpoint every N epochs
        filename='{epoch}-{val_loss:.4f}',
        monitor='val_loss',  # Monitor validation loss
        mode='min',  # Lower val_loss is better
    )

    # Setup loggers
    csv_logger = CSVLogger(
        save_dir=output_dir,
        name="metrics_logs"
    )

    # tensorboard_logger = TensorBoardLogger(
    #     save_dir=output_dir,
    #     name="tensorboard_logs"
    # )

    # Update trainer config with additional hooks
    trainer_config.update({
        'check_val_every_n_epoch': exp_config['check_val_every_n_epoch'],
        'callbacks': [checkpoint_callback],
        'logger': [csv_logger],  # Hoặc thêm tensorboard_logger nếu dùng
    })

    # Setup data
    data_wrapper = DataLoaderWrapper(**data_config)
    data_wrapper.setup()

    # Setup model
    model_wrapper = ModelWrapper(**model_config)

    # Setup trainer (pass config để evaluate_with_thresholds dùng)
    trainer = TrainerWrapper(config=config, **trainer_config)  # Thêm config=config

    # Train model
    trainer.fit(model_wrapper, data_wrapper.train_dataloader(), data_wrapper.val_dataloader())

    # Sau training, evaluate với nhiều thresholds trên val set
    print("\nEvaluating model with multiple thresholds...")
    best_th, all_results = trainer.evaluate_with_thresholds(model_wrapper, data_wrapper.val_dataloader())
    
    # Optional: Print chi tiết results (hoặc lưu vào file/log)
    for th, metrics in all_results.items():
        print(f"Threshold {th:.2f}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-score: {metrics['f1']:.4f}")
        print(f"  Confusion Matrix: {metrics['confusion_matrix']}")
    
    # Optional: Lưu best threshold vào config hoặc file
    with open(os.path.join(output_dir, 'best_threshold.txt'), 'w') as f:
        f.write(f"Best threshold: {best_th} (based on {config['metrics'].get('metric_priority', 'f1')}: {all_results[best_th][config['metrics'].get('metric_priority', 'f1')]:.4f})")

if __name__ == "__main__":
    main()