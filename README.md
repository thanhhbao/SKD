# SKD: Skin Cancer Detection Framework

This framework provides a modular, configurable pipeline for skin cancer detection and other medical image classification tasks, with a focus on efficient metrics logging and experiment tracking.

## Project Structure

```
SKD/
├── __init__.py
├── main.py                # Main entry point for training
├── configs/               # Configuration files for experiments
│   └── convnextv2_tiny.yaml
├── datasets/              # Dataset implementations
│   ├── __init__.py
│   ├── registry.py        # Dataset registry for registration and loading
│   └── isic2016.py        # ISIC 2016 dataset implementation
├── models/                # Model implementations
│   ├── __init__.py
│   ├── registry.py        # Model registry for registration and loading
│   ├── convnextv2.py      # ConvNeXtV2 model implementation
│   └── model_wrapper.py   # Lightning module wrapper for models
├── preprocessor/          # Image preprocessing
│   ├── __init__.py
│   ├── registry.py
│   └── base_processor.py
├── loss/                  # Loss functions
│   ├── __init__.py
│   ├── registry.py
│   └── base_losses.py
├── utils/                 # Utility modules
│   ├── __init__.py
│   ├── base_registry.py   # Base registry implementation
│   ├── config.py          # Configuration utilities
│   ├── data_loader.py     # DataLoader wrapper
│   ├── metrics.py         # Metrics logging system
│   └── trainer.py         # Trainer wrapper
└── outputs/               # Experiment outputs (created during training)
```

## Installation

### Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)

### Setup

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd SKD
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running Experiments

To train a model, use the `main.py` script with a configuration file:

```bash
python main.py --config configs/convnextv2_tiny.yaml
```

Arguments:
- `-c, --config`: Path to the configuration file (YAML format)
- `-e, --exp_name`: Experiment name (overrides the name in config file)

### Configuration System

The framework uses YAML configuration files to control all aspects of training. The configuration is divided into sections:

1. **experiment**: General experiment settings
   - `name`: Experiment name
   - `base_output_dir`: Directory to save outputs
   - `check_val_every_n_epoch`: Validation frequency
   - `save_top_k`: Number of best checkpoints to save

2. **model**: Model configuration
   - `name`: Model name (registered in model registry)
   - `loss_name`: Loss function name
   - `lr`: Learning rate
   - `device`: Training device (cuda/cpu)

3. **data**: Dataset configuration
   - `dataset_name`: Dataset name (registered in dataset registry)
   - `num_classes`: Number of classes
   - `*_batch_size`: Batch sizes for train/val/test
   - `test_size`: Validation split ratio
   - `dataset_kwargs`: Dataset-specific parameters
     - `preprocessor`: Image preprocessing configuration
     - Dataset-specific paths and column names

4. **trainer**: PyTorch Lightning Trainer settings
   - `seed`: Random seed
   - `devices`: Number of GPUs
   - `accelerator`: Accelerator type (gpu/cpu)
   - `max_epochs`: Maximum number of epochs
   - `precision`: Training precision (fp16/fp32)

5. **metrics**: Metrics configuration
   - `task`: Task type (binary/multiclass/multilabel)
   - `average`: Averaging method for metrics (macro/micro/weighted)
   - `include_precision`: Whether to include precision metrics
   - `include_recall`: Whether to include recall metrics

### Creating a Custom Experiment

1. **Create a configuration file**:
   - Copy an existing config file from `configs/` and modify it
   - Adjust dataset paths, model parameters, and training settings

2. **Run the experiment**:
   ```bash
   python main.py --config path/to/your/config.yaml
   ```

### Using Custom Datasets

1. Create a new dataset class in the `datasets/` directory
2. Register your dataset with this decorator `@DATASET_REGISTRY.register()`
3. Import your dataset to `__init__.py` file to initialize the module when running
4. Use your dataset in the configuration file by setting `data.dataset_name` to your registered name

Example dataset registration:
```python
from .registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
def load_my_custom_dataset(*args, **kwargs):
  dataset = {}
  dataset['train'] = MyCustomDataset(is_train=True, *args, **kwargs)
  dataset['test'] = MyCustomDataset(is_train=False, *args, **kwargs)
  return dataset 
```

### Using Custom Models

1. Create a new model in the `models/` directory
2. Register your model with this decorator `@BACKBONE.registor()`
3. Import your model to `__init__.py` file to initialize the module when running
4. Use your model in the configuration file by setting `model.name` to your registered name

Example model registration:
```python
from .registry import BACKBONE

@BACKBONE.register()
class Model(nn.Module):
  """
    Model implementation example.
  """
  def __init__(self, model_name: str, num_labels: int):
    super().__init__()
    # Implementation

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Implementation
```

## Metrics Logging
Metrics are automatically logged to:
- CSV files in the experiment output directory
- TensorBoard logs (can be viewed with `tensorboard --logdir outputs/experiment_name`) -> Currently deprecated due to increasing processing time.

## Checkpointing

The framework automatically saves model checkpoints based on validation loss. To resume training from a checkpoint:

1. Modify your configuration file to point to the checkpoint
2. Run training with the updated configuration

## Dataset Support

Currently supported datasets:
- ISIC 2016 (Skin Lesion Classification)

To use the ISIC 2016 dataset:
1. Download the dataset from the official ISIC Challenge website
2. Update the paths in your configuration file to point to the dataset location

## Extending the Framework

The framework is designed to be modular and extensible:

1. **Adding metrics**: Extend the `MetricsLogger` class in `utils/metrics.py`
2. **Adding models**: Create a new model file and register it in the model registry
3. **Adding datasets**: Create a new dataset file and register it in the dataset registry
4. **Adding preprocessors**: Create a new preprocessor and register it in the preprocessor registry

## License
[MIT License]