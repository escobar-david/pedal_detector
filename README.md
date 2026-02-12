# Pedal Detector (YOLOv8)

Object detection pipeline for identifying guitar pedals in pedalboard photos using YOLOv8n.

This project focuses on a practical end-to-end ML workflow in Python: dataset preparation, reproducible training, CLI inference, and model export.

## Why This Project

- Real-world computer vision task on custom data.
- Complete workflow from raw labels to inference artifacts.
- Lightweight model choice for fast inference.

## Dataset Summary

- Total labeled images: 164 (`images/`: 165 files, `labels/`: 164 files).
- Train/val split: 139 / 25 (84.8% / 15.2%).
- Class setup: single class `guitar_pedal` (YOLO TXT detection format).
- Labeled objects: 1,652 total (1,413 train, 239 val), about 10.1 pedals per labeled image.
- Raw training data (`images/`, `labels/`, `datasets/`) remains private due to licensing constraints.

## Results Snapshot

Reference validation run: YOLOv8n, 50 epochs, batch 8, imgsz 640, seed 0, CPU.

| Checkpoint | Precision | Recall | F1 | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|---:|
| Baseline (epoch 1) | 0.7358 | 0.7082 | 0.7217 | 0.7543 | 0.4421 |
| Best (epoch 39) | 0.9407 | 0.9411 | 0.9409 | 0.9751 | 0.7343 |
| Final (epoch 50) | 0.9042 | 0.9596 | 0.9311 | 0.9759 | 0.7157 |

Baseline to best improvement:
- mAP@50-95: +0.2922 (+66.1%)
- F1: +0.2192 (+30.4%)

## Example Prediction

| Original | YOLOv8 Prediction |
|---|---|
| ![Original pedalboard](my_pedals.jpg) | ![Predicted pedalboard](runs/detect/predict/my_pedals.jpg) |

Generated with: `python predict.py --source my_pedals.jpg --weights runs/detect/pedal_detector/weights/best.pt --conf 0.25 --iou 0.45 --save`.

## Quick Start

```bash
git clone https://github.com/<your-user>/pedal_detector.git
cd pedal_detector
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Download pretrained weights from GitHub Releases:

```bash
curl -L -o best.pt https://github.com/escobar-david/pedal_detector/releases/download/yolo-pedalboards/best.pt
python predict.py --source my_pedals.jpg --weights best.pt --save
```

## Reproduce Training

Reference run used for the metrics above:

```bash
python split_dataset.py --source-images images --source-labels labels --output-dir datasets --train-ratio 0.85 --seed 42
python train.py --data data.yaml --epochs 50 --batch 8 --imgsz 640 --device cpu --seed 0 --project runs/detect --name pedal_detector
python predict.py --source my_pedals.jpg --weights runs/detect/pedal_detector/weights/best.pt --save
```

Reference run training time: 7271.99 seconds (~2h 1m 12s).

CPU-only training:

```bash
python train.py --device cpu
```

## CLI Usage

```bash
python train.py --help
python predict.py --help
python split_dataset.py --help
python interactive.py --help
```

### Common CLI examples

Train with explicit run naming and seed:

```bash
python train.py --data data.yaml --epochs 50 --batch 16 --seed 123 --project runs/detect --name exp_seed123
```

Predict on CPU with custom thresholds:

```bash
python predict.py --source my_pedals.jpg --weights runs/detect/pedal_detector/weights/best.pt --device cpu --conf 0.3 --iou 0.45 --save
```

Run interactive mode on CPU and explicit weights:

```bash
python interactive.py --device cpu --weights runs/detect/pedal_detector/weights/best.pt
```

Split data with custom output and deterministic shuffle:

```bash
python split_dataset.py --source-images images --source-labels labels --output-dir datasets --train-ratio 0.9 --seed 7
```

### Notes

- `predict.py` now exits with a non-zero code when `--source` or `--weights` is invalid.
- `train.py` supports reproducible runs via `--seed` and portable output paths via `--project` and `--name`.

### Exit codes

- `train.py`: returns `0` on successful execution.
- `split_dataset.py`: returns `0` on successful split, non-zero for invalid `--train-ratio` or missing source directories.
- `predict.py`: returns `0` on successful inference, non-zero when input source or weights are missing/invalid.
- `interactive.py`: returns `0` on normal session exit (`q`/`quit`/`exit`), non-zero if weights cannot be resolved at startup.

## Project Structure

```text
pedal_detector/
|-- data.yaml
|-- split_dataset.py
|-- train.py
|-- predict.py
|-- interactive.py
|-- requirements.txt
|-- images/                # raw images (local/private by default)
|-- labels/                # raw labels (local/private by default)
|-- datasets/              # generated train/val split
`-- runs/                  # training and inference outputs
```

## Design Choices

- Model: YOLOv8n for speed and compact size.
- Transfer learning from pretrained weights.
- Augmentation tuned for small dataset:
- HSV jitter
- Horizontal flips
- Moderate rotation (+/-10 degrees)
- Mosaic and mixup

## Limitations

- Single class only: `guitar_pedal`.
- Small dataset size may reduce generalization.
- No instance segmentation or pedal-type classification yet.

## Roadmap

- Multi-class labels by pedal type.
- Structured error analysis and evaluation reports.
- Dockerized inference service.

## Dataset and Rights

Before publishing images/labels publicly, confirm you have redistribution rights for every photo and annotation.
Raw training data (`images/`, `labels/`, `datasets/`) is kept private in this project due to licensing constraints.

## License

MIT. See `LICENSE`.
