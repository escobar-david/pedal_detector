# Guitar Pedal Detector

YOLOv8 Nano object detection model for detecting guitar pedals in images.

## Dataset

- **Images**: 165 images
- **Annotations**: 1,652 bounding boxes (~10 pedals per image)
- **Format**: YOLO format (single class: `guitar_pedal`)

## Installation

```bash
pip install -r requirements.txt
```

### GPU Support

For GPU acceleration, ensure you have CUDA installed. The training script will automatically use GPU if available.

For CPU-only training, use:
```bash
python train.py --device cpu
```

## Project Structure

```
pedal_detector/
├── images/              # Original images (backup)
├── labels/              # Original labels (backup)
├── datasets/            # Split dataset (created by split_dataset.py)
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   └── val/
│       ├── images/
│       └── labels/
├── runs/                # Training outputs
├── data.yaml            # Dataset configuration
├── train.py             # Training script
├── predict.py           # Inference script
├── split_dataset.py     # Dataset splitting utility
└── requirements.txt     # Python dependencies
```

## Usage

### 1. Split the Dataset

First, split your dataset into training and validation sets:

```bash
python split_dataset.py
```

This will:
- Create an 85/15 train/val split
- Copy files to `datasets/train/` and `datasets/val/`
- Report any images missing labels

### 2. Train the Model

```bash
python train.py
```

#### Training Options

```bash
python train.py --epochs 100 --batch 16 --imgsz 640 --device 0
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 100 | Number of training epochs |
| `--batch` | 16 | Batch size (reduce if OOM) |
| `--imgsz` | 640 | Input image size |
| `--device` | 0 | GPU device or "cpu" |
| `--workers` | 8 | Data loader workers |
| `--patience` | 20 | Early stopping patience |
| `--resume` | - | Resume from checkpoint |

### 3. Run Inference

```bash
# Single image
python predict.py --source path/to/image.jpg --save

# Directory of images
python predict.py --source path/to/images/ --save

# Show results
python predict.py --source image.jpg --show

# Adjust confidence threshold
python predict.py --source image.jpg --conf 0.5 --save
```

#### Inference Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | required | Image, directory, or video |
| `--weights` | best.pt | Path to model weights |
| `--conf` | 0.25 | Confidence threshold |
| `--iou` | 0.45 | IoU threshold for NMS |
| `--save` | - | Save annotated results |
| `--show` | - | Display results |
| `--save-txt` | - | Save labels as .txt |
| `--save-crop` | - | Save cropped detections |

### 4. Export Model

For deployment, export the model to other formats:

```python
from predict import export_model

# Export to ONNX
export_model(format='onnx')

# Export to TorchScript
export_model(format='torchscript')

# Export to TFLite
export_model(format='tflite')
```

## Training Details

### Model
- **Architecture**: YOLOv8 Nano (fastest, smallest)
- **Pretrained**: COCO weights for transfer learning

### Augmentation
Optimized for small dataset (165 images):
- Color jittering (HSV)
- Horizontal flip (no vertical - pedals have orientation)
- Rotation (moderate, ±10°)
- Scale variation
- Mosaic augmentation
- MixUp

### Hyperparameters
- Optimizer: AdamW
- Initial LR: 0.001
- Early stopping: 20 epochs patience
- Image size: 640x640

## Expected Performance

With this dataset size and configuration, expect:
- **mAP50**: 0.7-0.9
- **mAP50-95**: 0.5-0.7

Performance depends on image quality and annotation consistency.

## Output

Training outputs are saved to `runs/detect/pedal_detector/`:
- `weights/best.pt` - Best model weights
- `weights/last.pt` - Last epoch weights
- Training curves and metrics
- Validation predictions
