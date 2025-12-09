# Anomaly OV Fine-tuning Module

This module provides a complete fine-tuning pipeline for the Anomaly OV model with AnyRes preprocessing.

## Structure

```
finetuning/
├── __init__.py          # Module exports
├── config.py            # Configuration dataclass and CLI argument parsing
├── dataset.py           # AnomalyDataset and data loading utilities
├── losses.py            # Loss functions (AnomalyStage1Loss, FocalLoss, etc.)
├── trainer.py           # Trainer class with training loop
├── evaluation.py        # Metrics computation and inference utilities
├── visualization.py     # Visualization functions
└── train.py             # Main training script
```

## Usage

### Basic Training

```bash
# From the project root directory
python train_anomaly_ov.py
```

Or run as a module:

```bash
python -m finetuning.train
```

### Resume from Checkpoint

```bash
python train_anomaly_ov.py --resume --resume-path ./checkpoints/checkpoint_epoch_1.pt
```

### Custom Settings

```bash
python train_anomaly_ov.py \
    --epochs 10 \
    --batch-size 8 \
    --lr 5e-5 \
    --train-samples 500 \
    --eval-samples 100 \
    --test-samples 100
```

### All Options

```bash
python train_anomaly_ov.py --help
```

Available options:

| Option | Default | Description |
|--------|---------|-------------|
| `--data-root` | `./finetune_dataset` | Root directory for dataset |
| `--initial-checkpoint` | `./zs_checkpoint.pt` | Path to initial model checkpoint |
| `--vision-tower` | `google/siglip-so400m-patch14-384` | Vision tower name |
| `--batch-size` | `4` | Batch size |
| `--epochs` | `4` | Number of epochs |
| `--lr` | `1e-4` | Learning rate |
| `--weight-decay` | `1e-5` | Weight decay |
| `--loss-tau` | `1.0` | Temperature for BCE loss |
| `--loss-pos-weight` | `1.5` | Positive class weight |
| `--loss-margin` | `0.0` | Margin for loss |
| `--checkpoint-dir` | `./checkpoints` | Directory to save checkpoints |
| `--save-every` | `1` | Save checkpoint every N epochs |
| `--resume` | `False` | Resume from checkpoint |
| `--resume-path` | `None` | Path to checkpoint to resume from |
| `--train-samples` | `250` | Max training samples (0 for all) |
| `--eval-samples` | `50` | Max eval samples (0 for all) |
| `--test-samples` | `50` | Max test samples (0 for all) |
| `--no-visualize` | `False` | Disable visualization during training |
| `--vis-samples` | `8` | Number of visualization samples |
| `--log-dir` | `./logs` | TensorBoard log directory |
| `--config-json` | `./config.json` | Path to model config JSON |

## Programmatic Usage

You can also use the module programmatically:

```python
from finetuning import Config, Trainer, AnomalyDataset, compute_metrics
from finetuning.dataset import create_dataloaders
from custom_anomaly_detector import OVAnomalyDetector

# Create configuration
config = Config(
    num_epochs=10,
    batch_size=8,
    learning_rate=5e-5,
    resume_from_checkpoint=True,
    resume_checkpoint_path="./checkpoints/checkpoint_epoch_1.pt"
)

# Load model
detector, image_processor = OVAnomalyDetector.load_from_checkpoint(
    config.initial_checkpoint,
    device=config.device
)

# Create dataloaders
train_loader, eval_loader, test_loader, train_ds, eval_ds, test_ds = \
    create_dataloaders(config, image_processor)

# Create trainer and train
trainer = Trainer(
    model=detector,
    config=config,
    train_loader=train_loader,
    eval_loader=eval_loader,
    test_dataset=test_ds
)
history = trainer.train()

# Evaluate
metrics = compute_metrics(detector, test_loader, config.device)
print(f"Test AUC: {metrics['auc']:.4f}")
```

## Dataset Structure

The dataset should be organized as follows:

```
finetune_dataset/
├── train/
│   ├── inpainted/     # Inpainted images (anomalies)
│   ├── masks/         # Binary masks for anomaly regions
│   └── COCO_real/     # Original real images (no anomalies)
├── eval/
│   ├── inpainted/
│   ├── masks/
│   └── COCO_real/
└── test/
    ├── inpainted/
    ├── masks/
    └── COCO_real/
```

## Output

After training, you'll find:

- `checkpoints/best_model.pt` - Best model based on eval loss
- `checkpoints/checkpoint_epoch_N.pt` - Checkpoints for each epoch
- `checkpoints/training_config.json` - Training configuration and results
- `checkpoints/training_history.png` - Loss and accuracy curves
- `checkpoints/roc_curve_test.png` - ROC curve on test set
- `checkpoints/predictions_*.png` - Sample predictions with anomaly maps
- `logs/` - TensorBoard logs

## Monitoring Training

View TensorBoard logs:

```bash
tensorboard --logdir=./logs
```
